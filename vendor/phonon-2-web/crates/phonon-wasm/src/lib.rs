//! Phonon-2 for the browser.
//!
//! Everything here runs inside a dedicated Web Worker (see `web/phonon-worker.js`): the CPU encoder blocks while rayon's
//! workers compute, which the main thread may not do. JS API:
//!
//! ```js
//! await init(); await initThreadPool(n);          // threads build only
//! const p = Phonon.load(bytes, configOrNull, (stage, done, total) => ...);
//! await p.enableGpu();                            // optional WebGPU encoder; resolves to the adapter description
//! p.enableGl();                                   // optional WebGL2 encoder (browsers without WebGPU)
//! p.useBackend('cpu' | 'webgpu' | 'webgl');
//! const json = await p.transcribe(float32Audio16k);
//! const ev = await p.liveFeed(float32Block);      // {"partial": "...", "finals": ["..."]}
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use phonon_core::engine::WindowOut;
use phonon_core::live::{LiveAction, LiveSegmenter};
use phonon_core::{segment, Engine, Model, Transcript};
use wasm_bindgen::prelude::*;

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[derive(Clone, Copy, PartialEq)]
enum Backend {
    Cpu,
    WebGpu,
    WebGl,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::WebGpu => "webgpu",
            Backend::WebGl => "webgl",
        }
    }
}

struct Inner {
    engine: Engine,
    #[cfg(feature = "gpu")]
    gpu: Option<phonon_gpu::GpuEncoder>,
    #[cfg(feature = "webgl")]
    gl: Option<phonon_gl::GlEncoder>,
    backend: Backend,
    live: LiveSegmenter,
    /// time spent in the last call, ms: (features, encoder, decoder)
    last: (f64, f64, f64),
}

impl Inner {
    /// Projected encoder output for one window on the active backend.
    async fn encode(this: &Rc<RefCell<Inner>>, audio: &[f32]) -> Result<(Vec<f32>, usize), JsError> {
        let t0 = now_ms();
        let (mel, frames) = this.borrow().engine.features(audio);
        let t1 = now_ms();
        let backend = this.borrow().backend;
        #[cfg(feature = "webgl")]
        if backend == Backend::WebGl {
            let mut me = this.borrow_mut();
            let out = me.gl.as_mut().ok_or_else(|| JsError::new("WebGL not initialised"))?.encode(&mel, frames).map_err(|e| JsError::new(&format!("WebGL encode: {e}")))?;
            me.last.0 += t1 - t0;
            me.last.1 += now_ms() - t1;
            return Ok(out);
        }
        #[cfg(feature = "gpu")]
        {
            if backend == Backend::WebGpu {
                // take the encoder out so no RefCell borrow lives across the await
                let mut gpu = this.borrow_mut().gpu.take().ok_or_else(|| JsError::new("GPU not initialised"))?;
                let r = gpu.encode(&mel, frames).await;
                this.borrow_mut().gpu = Some(gpu);
                let out = r.map_err(|e| JsError::new(&format!("GPU encode: {e}")))?;
                let mut me = this.borrow_mut();
                me.last.0 += t1 - t0;
                me.last.1 += now_ms() - t1;
                return Ok(out);
            }
        }
        let mut me = this.borrow_mut();
        let me = &mut *me;
        let out = phonon_core::encoder::encode(&me.engine.model, &mut me.engine.pos, &mel, frames);
        me.last.0 += t1 - t0;
        me.last.1 += now_ms() - t1;
        Ok(out)
    }

    async fn window(this: &Rc<RefCell<Inner>>, audio: &[f32]) -> Result<Vec<phonon_core::decoder::Token>, JsError> {
        if segment::is_silent(audio) {
            return Ok(Vec::new());
        }
        let (enc, t) = Self::encode(this, audio).await?;
        let t0 = now_ms();
        let toks = this.borrow().engine.decode(&enc, t);
        this.borrow_mut().last.2 += now_ms() - t0;
        Ok(toks)
    }

    async fn transcribe(this: &Rc<RefCell<Inner>>, audio: &[f32]) -> Result<Transcript, JsError> {
        let mut wins = Vec::new();
        if !segment::is_silent(audio) {
            for (a, b) in segment::plan(audio) {
                let tokens = Self::window(this, &audio[a..b]).await?;
                wins.push(WindowOut { start: a, end: b, tokens });
            }
        }
        Ok(Transcript::from_windows(&this.borrow().engine.model.vocab, audio.len(), &wins))
    }
}

#[wasm_bindgen]
pub struct Phonon {
    inner: Rc<RefCell<Inner>>,
}

#[wasm_bindgen]
impl Phonon {
    /// Load `phonon-2.bps.tar.zst` (or a bare `model.fermion` with `config` = the bytes of its `config.json`).
    /// `progress(stage, done, total)` is called while the weights are read.
    pub fn load(bytes: &[u8], config: Option<Vec<u8>>, progress: Option<js_sys::Function>) -> Result<Phonon, JsError> {
        console_hook();
        let mut last = -1i64;
        let mut cb = |stage: &str, done: u64, total: u64| {
            let pct = if total > 0 { (done * 100 / total) as i64 } else { -1 };
            if pct != last {
                last = pct;
                if let Some(f) = &progress {
                    let _ = f.call3(&JsValue::NULL, &JsValue::from_str(stage), &JsValue::from_f64(done as f64), &JsValue::from_f64(total as f64));
                }
            }
        };
        let model = Model::load(bytes, config.as_deref(), &mut cb).map_err(|e| JsError::new(&e.0))?;
        Ok(Phonon {
            inner: Rc::new(RefCell::new(Inner {
                engine: Engine::new(model),
                #[cfg(feature = "gpu")]
                gpu: None,
                #[cfg(feature = "webgl")]
                gl: None,
                backend: Backend::Cpu,
                live: LiveSegmenter::new(),
                last: (0.0, 0.0, 0.0),
            })),
        })
    }

    /// JSON: weights in memory, threads, backend.
    pub fn info(&self) -> String {
        let me = self.inner.borrow();
        serde_json::json!({
            "weight_bytes": me.engine.model.bytes(),
            "pos_cache_bytes": me.engine.pos.bytes(),
            "threads": phonon_core::par::threads(),
            "simd": cfg!(target_feature = "simd128"),
            "webgpu_built": cfg!(feature = "gpu"),
            "webgl_built": cfg!(feature = "webgl"),
            "backend": me.backend.name(),
        })
        .to_string()
    }

    /// Initialise WebGPU and upload the encoder (once). Resolves to the adapter description.
    #[wasm_bindgen(js_name = enableGpu)]
    pub fn enable_gpu(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            #[cfg(feature = "gpu")]
            {
                if inner.borrow().gpu.is_none() {
                    let t0 = now_ms();
                    let log: Box<dyn Fn(&str)> = Box::new(move |stage: &str| console_info(&format!("phonon gpu: {stage} at {:.0} ms", now_ms() - t0)));
                    let opts = phonon_gpu::Options { log: Some(log), ..Default::default() };
                    let enc = {
                        let me = inner.borrow();
                        phonon_gpu::GpuEncoder::new(&me.engine.model, opts).await
                    }
                    .map_err(|e| JsValue::from(JsError::new(&format!("WebGPU: {e}"))))?;
                    inner.borrow_mut().gpu = Some(enc);
                }
                let mut me = inner.borrow_mut();
                me.backend = Backend::WebGpu;
                Ok(JsValue::from_str(&me.gpu.as_ref().unwrap().adapter_info()))
            }
            #[cfg(not(feature = "gpu"))]
            {
                let _ = inner;
                Err(JsValue::from(JsError::new("this build has no GPU support")))
            }
        })
    }

    /// Initialise WebGL2 (an OffscreenCanvas context in this worker) and upload the encoder (once). Returns the
    /// renderer description and makes WebGL the active backend.
    #[wasm_bindgen(js_name = enableGl)]
    pub fn enable_gl(&self) -> Result<String, JsError> {
        #[cfg(feature = "webgl")]
        {
            let mut me = self.inner.borrow_mut();
            if me.gl.is_none() {
                let enc = phonon_gl::GlEncoder::new(&me.engine.model).map_err(|e| JsError::new(&format!("WebGL: {e}")))?;
                me.gl = Some(enc);
            }
            me.backend = Backend::WebGl;
            Ok(me.gl.as_ref().unwrap().adapter_info())
        }
        #[cfg(not(feature = "webgl"))]
        Err(JsError::new("this build has no WebGL support"))
    }

    /// Drop a GPU backend's device state and weights ("webgpu" or "webgl"), freeing its GPU memory. If it was active,
    /// the CPU takes over.
    pub fn release(&self, name: &str) {
        let mut me = self.inner.borrow_mut();
        match name {
            #[cfg(feature = "gpu")]
            "webgpu" => {
                me.gpu = None;
                if me.backend == Backend::WebGpu {
                    me.backend = Backend::Cpu;
                }
            }
            #[cfg(feature = "webgl")]
            "webgl" => {
                me.gl = None;
                if me.backend == Backend::WebGl {
                    me.backend = Backend::Cpu;
                }
            }
            _ => {}
        }
    }

    /// Switch between initialised backends: "cpu", "webgpu" or "webgl".
    #[wasm_bindgen(js_name = useBackend)]
    pub fn use_backend(&self, name: &str) -> Result<(), JsError> {
        let mut me = self.inner.borrow_mut();
        let b = match name {
            "cpu" => Backend::Cpu,
            #[cfg(feature = "gpu")]
            "webgpu" if me.gpu.is_some() => Backend::WebGpu,
            #[cfg(feature = "webgl")]
            "webgl" if me.gl.is_some() => Backend::WebGl,
            other => return Err(JsError::new(&format!("backend {other:?} is not initialised"))),
        };
        me.backend = b;
        Ok(())
    }

    /// Transcribe 16 kHz mono audio of any length. Resolves to JSON: text, words, tokens, windows, timings.
    pub fn transcribe(&self, audio: Vec<f32>) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            inner.borrow_mut().last = (0.0, 0.0, 0.0);
            let t0 = now_ms();
            let tr = Inner::transcribe(&inner, &audio).await.map_err(JsValue::from)?;
            let total = now_ms() - t0;
            let me = inner.borrow();
            let mut j = tr.to_json();
            j["timings_ms"] = serde_json::json!({"total": total, "features": me.last.0, "encoder": me.last.1, "decoder": me.last.2});
            j["backend"] = serde_json::json!(me.backend.name());
            Ok(JsValue::from_str(&j.to_string()))
        })
    }

    /// Live dictation: feed microphone samples (16 kHz mono, any block size). Resolves to JSON
    /// `{"partial": text|null, "finals": [text...], "segment_s": seconds in flight}`; superseded partials are skipped.
    #[wasm_bindgen(js_name = liveFeed)]
    pub fn live_feed(&self, samples: Vec<f32>) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let actions = inner.borrow_mut().live.feed(&samples);
            live_run(&inner, actions).await
        })
    }

    /// End the live session: finalise the segment in flight.
    #[wasm_bindgen(js_name = liveFinish)]
    pub fn live_finish(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let actions: Vec<LiveAction> = inner.borrow_mut().live.finish().into_iter().collect();
            live_run(&inner, actions).await
        })
    }

    /// Drop the in-flight live segment.
    #[wasm_bindgen(js_name = liveReset)]
    pub fn live_reset(&self) {
        self.inner.borrow_mut().live = LiveSegmenter::new();
    }
}

async fn live_run(inner: &Rc<RefCell<Inner>>, actions: Vec<LiveAction>) -> Result<JsValue, JsValue> {
    let last_partial = actions.iter().rposition(|a| matches!(a, LiveAction::Partial(_)));
    let last_final = actions.iter().rposition(|a| matches!(a, LiveAction::Final(_)));
    let mut finals = Vec::new();
    let mut partial = None;
    inner.borrow_mut().last = (0.0, 0.0, 0.0);
    for (i, a) in actions.into_iter().enumerate() {
        match a {
            LiveAction::Final(audio) => {
                let toks = Inner::window(inner, &audio).await.map_err(JsValue::from)?;
                let text = inner.borrow().engine.text(&toks);
                if !text.is_empty() {
                    finals.push(text);
                }
            }
            // a partial is only worth decoding if nothing newer supersedes it
            LiveAction::Partial(audio) if Some(i) == last_partial && last_final.is_none_or(|f| f < i) => {
                let toks = Inner::window(inner, &audio).await.map_err(JsValue::from)?;
                let text = inner.borrow().engine.text(&toks);
                if !text.is_empty() {
                    partial = Some(text);
                }
            }
            LiveAction::Partial(_) => {}
        }
    }
    let me = inner.borrow();
    Ok(JsValue::from_str(
        &serde_json::json!({
            "partial": partial,
            "finals": finals,
            "segment_s": me.live.segment_seconds(),
            "decode_ms": me.last.0 + me.last.1 + me.last.2,
        })
        .to_string(),
    ))
}

fn console_hook() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            console_error(&format!("phonon panic: {info}"));
        }));
    });
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = info)]
    fn console_info(s: &str);
}
