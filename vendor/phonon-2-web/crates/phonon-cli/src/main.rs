//! `phonon-rs`: native test bench for the port.
//!
//!   phonon-rs check  <model> <testdata-dir>     compare every stage with the golden reference (tools/make_reference.py)
//!   phonon-rs run    <model> <wav>...           transcribe wav files (16 kHz mono)
//!   phonon-rs bench  <model> <wav> [reps]       time the stages
//!
//! `<model>` is `phonon-2.bps.tar.zst` or a directory holding `model.fermion` + `config.json`.
//! `--gpu` (with the `gpu` feature) runs the encoder on the GPU; `--threads N` sets the CPU pool.

use std::path::{Path, PathBuf};
use std::time::Instant;

use phonon_core::model::PRED;
use phonon_core::{encoder, Engine, Model, SAMPLE_RATE};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let gpu = take_flag(&mut args, "--gpu");
    if let Some(n) = take_opt(&mut args, "--threads") {
        rayon::ThreadPoolBuilder::new().num_threads(n.parse().expect("--threads N")).build_global().unwrap();
    }
    if args.len() < 2 {
        eprintln!("usage: phonon-rs check|run|bench <model> ...");
        std::process::exit(2);
    }
    let started = Instant::now();
    let model = load_model(Path::new(&args[1]));
    eprintln!(
        "loaded in {:.2}s, {:.0} MB of weights, {} threads",
        started.elapsed().as_secs_f64(),
        model.bytes() as f64 / 1e6,
        rayon::current_num_threads()
    );
    let mut engine = Engine::new(model);
    match args[0].as_str() {
        "check" => check(&mut engine, Path::new(&args[2]), gpu),
        "run" => {
            for w in &args[2..] {
                let audio = read_wav(Path::new(w));
                let t = Instant::now();
                let tr = if gpu { gpu_transcribe(&mut engine, &audio) } else { engine.transcribe(&audio) };
                let dt = t.elapsed().as_secs_f64();
                println!("{w}: [{:.2}s audio, {:.2}s, {:.1}x] {}", tr.audio_seconds, dt, tr.audio_seconds / dt, tr.text);
            }
        }
        #[cfg(feature = "gpu")]
        "gpuprof" => {
            let audio = read_wav(Path::new(&args[2]));
            let (mel, frames) = engine.features(&audio);
            gpu_encode(&mut engine, &mel, frames);
            gpu_glue::GPU.with(|g| {
                for (k, t, n) in g.borrow_mut().as_mut().unwrap().profile(&mel, frames) {
                    println!("{k:<18} {:>8.2} ms  x{n}", t * 1e3);
                }
            });
        }
        "bench" => bench(&mut engine, Path::new(&args[2]), args.get(3).map_or(3, |s| s.parse().unwrap()), gpu),
        other => panic!("unknown command {other}"),
    }
}

fn take_flag(args: &mut Vec<String>, f: &str) -> bool {
    let n = args.len();
    args.retain(|a| a != f);
    args.len() != n
}

fn take_opt(args: &mut Vec<String>, f: &str) -> Option<String> {
    let i = args.iter().position(|a| a == f)?;
    let v = args.get(i + 1).cloned();
    args.drain(i..(i + 2).min(args.len()));
    v
}

fn load_model(p: &Path) -> Model {
    let mut last = 0u64;
    let mut prog = |stage: &str, done: u64, total: u64| {
        if total > 0 && done * 10 / total != last {
            last = done * 10 / total;
            eprint!("\r{stage} {}%   ", last * 10);
        }
    };
    let m = if p.is_dir() {
        let bytes = std::fs::read(p.join("model.fermion")).expect("model.fermion");
        let cfg = std::fs::read(p.join("config.json")).expect("config.json");
        Model::load(&bytes, Some(&cfg), &mut prog)
    } else {
        let bytes = std::fs::read(p).expect("model file");
        Model::load(&bytes, None, &mut prog)
    };
    eprintln!();
    m.unwrap_or_else(|e| panic!("load: {e}"))
}

fn read_wav(p: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let spec = r.spec();
    assert_eq!(spec.sample_rate as usize, SAMPLE_RATE, "{}: need 16 kHz", p.display());
    let ch = spec.channels as usize;
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|v| v.unwrap() as f32 * scale).collect()
        }
        hound::SampleFormat::Float => r.samples::<f32>().map(|v| v.unwrap()).collect(),
    };
    s.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect()
}

fn read_f32(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// (max |a-b|, max |b|, relative RMS error)
fn diff(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut mx = 0f32;
    let mut mb = 0f32;
    let (mut se, mut sb) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        mx = mx.max((x - y).abs());
        mb = mb.max(y.abs());
        se += ((x - y) as f64).powi(2);
        sb += (*y as f64).powi(2);
    }
    (mx, mb, (se / sb.max(1e-30)).sqrt() as f32)
}

fn check(engine: &mut Engine, td: &Path, gpu: bool) {
    let refs: serde_json::Value = serde_json::from_slice(&std::fs::read(td.join("ref/reference.json")).unwrap()).unwrap();
    let mut names: Vec<_> = refs.as_object().unwrap().keys().cloned().collect();
    names.sort();
    let (mut same, mut total) = (0, 0);
    for name in names {
        let r = &refs[&name];
        let audio = read_wav(&td.join(format!("{name}.wav")));
        let rp = |s: &str| -> PathBuf { td.join("ref").join(format!("{name}.{s}.bin")) };
        let (mel, frames) = engine.features(&audio);
        let dm = diff(&mel, &read_f32(&rp("mel")));
        let t0 = Instant::now();
        let (enc, t) = if gpu {
            gpu_encode(engine, &mel, frames)
        } else {
            let (sub, l0, fin, t) = encoder::encode_debug(&engine.model, &mut engine.pos, &mel, frames);
            let ds = diff(&sub, &read_f32(&rp("sub")));
            let dl = diff(&l0, &read_f32(&rp("l0")));
            let de = diff(&fin, &read_f32(&rp("enc")));
            println!("{name}: mel rel {:.1e} | sub rel {:.1e} | layer0 rel {:.1e} | enc rel {:.1e} (max {:.3} of {:.1})", dm.2, ds.2, dl.2, de.2, de.0, de.1);
            let mut proj = vec![0f32; t * PRED];
            phonon_core::gemm::gemm(&fin, t, &engine.model.enc_proj, Some(&engine.model.enc_proj_b), &mut proj);
            (proj, t)
        };
        let enc_s = t0.elapsed().as_secs_f64();
        let dp = diff(&enc, &read_f32(&rp("proj")));
        let t1 = Instant::now();
        let toks = engine.decode(&enc, t);
        let dec_s = t1.elapsed().as_secs_f64();
        let ids: Vec<u64> = toks.iter().map(|t| t.id as u64).collect();
        let ref_ids: Vec<u64> = r["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let text = engine.text(&toks);
        let ok = ids == ref_ids;
        total += 1;
        same += ok as usize;
        println!(
            "{name}: T={t} (ref {}) proj rel {:.1e} | encode {:.2}s decode {:.2}s | tokens {} {}",
            r["T"],
            dp.2,
            enc_s,
            dec_s,
            if ok { "IDENTICAL" } else { "DIFFER" },
            if ok { String::new() } else { format!("\n   ours: {text}\n   ref:  {}", r["text"].as_str().unwrap()) }
        );
    }
    println!("{same}/{total} clips token-identical to the fp32 reference");
}

fn bench(engine: &mut Engine, wav: &Path, reps: usize, gpu: bool) {
    let audio = read_wav(wav);
    let secs = audio.len() as f64 / SAMPLE_RATE as f64;
    for i in 0..reps {
        let t0 = Instant::now();
        let (mel, frames) = engine.features(&audio);
        let t1 = Instant::now();
        let (enc, t) = if gpu { gpu_encode(engine, &mel, frames) } else { encoder::encode(&engine.model, &mut engine.pos, &mel, frames) };
        let t2 = Instant::now();
        let toks = engine.decode(&enc, t);
        let t3 = Instant::now();
        phonon_core::prof::report();
        let tot = (t3 - t0).as_secs_f64();
        println!(
            "rep {i}: {secs:.1}s audio | mel {:.3}s encoder {:.3}s decoder {:.3}s | total {:.3}s = {:.1}x realtime | {} tokens",
            (t1 - t0).as_secs_f64(),
            (t2 - t1).as_secs_f64(),
            (t3 - t2).as_secs_f64(),
            tot,
            secs / tot,
            toks.len()
        );
    }
}

#[cfg(feature = "gpu")]
mod gpu_glue {
    use std::cell::RefCell;
    thread_local! { pub static GPU: RefCell<Option<phonon_gpu::GpuEncoder>> = const { RefCell::new(None) }; }
}

#[cfg(feature = "gpu")]
fn gpu_encode(engine: &mut Engine, mel: &[f32], frames: usize) -> (Vec<f32>, usize) {
    gpu_glue::GPU.with(|g| {
        let mut g = g.borrow_mut();
        if g.is_none() {
            let t = Instant::now();
            let enc = pollster::block_on(phonon_gpu::GpuEncoder::new(&engine.model, phonon_gpu::Options::default())).expect("gpu init");
            eprintln!("gpu: {} ({:.2}s to upload)", enc.adapter_info(), t.elapsed().as_secs_f64());
            *g = Some(enc);
        }
        pollster::block_on(g.as_mut().unwrap().encode(mel, frames)).expect("gpu encode")
    })
}

#[cfg(feature = "gpu")]
fn gpu_transcribe(engine: &mut Engine, audio: &[f32]) -> phonon_core::Transcript {
    let wins: Vec<_> = phonon_core::segment::plan(audio)
        .into_iter()
        .map(|(a, b)| {
            let w = &audio[a..b];
            let tokens = if phonon_core::segment::is_silent(w) {
                vec![]
            } else {
                let (mel, frames) = engine.features(w);
                let (enc, t) = gpu_encode(engine, &mel, frames);
                engine.decode(&enc, t)
            };
            phonon_core::engine::WindowOut { start: a, end: b, tokens }
        })
        .collect();
    phonon_core::Transcript::from_windows(&engine.model.vocab, audio.len(), &wins)
}

#[cfg(not(feature = "gpu"))]
fn gpu_encode(_: &mut Engine, _: &[f32], _: usize) -> (Vec<f32>, usize) {
    panic!("built without the `gpu` feature")
}

#[cfg(not(feature = "gpu"))]
fn gpu_transcribe(_: &mut Engine, _: &[f32]) -> phonon_core::Transcript {
    panic!("built without the `gpu` feature")
}
