use js_sys::{Float32Array, Object, Reflect, Uint32Array};
use phonon_core::encoder::{encoder_frames, pos_embedding};
use phonon_core::model::{Model, D_FF, D_MODEL, HEAD_DIM, N_HEADS, N_MELS, PRED, SUB_CH};
use phonon_core::tensor::{FvMatrix, QMatrix, CODE_NEG_HI, CODE_NEG_LO, CODE_POS_HI, CODE_POS_LO};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{OffscreenCanvas, WebGl2RenderingContext as Gl, WebGlFramebuffer, WebGlProgram, WebGlTexture, WebGlUniformLocation};

use crate::shaders;

const WIDTH: i32 = 1024;
pub const MAX_FRAMES: usize = 512;

#[derive(Clone)]
struct Tex {
    t: WebGlTexture,
}

struct Prog {
    p: WebGlProgram,
    ua: Option<WebGlUniformLocation>,
    ub: Option<WebGlUniformLocation>,
    uf: Option<WebGlUniformLocation>,
    nsamp: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum P {
    GemmFv,
    GemmDense,
    LnStats,
    LnApply,
    Scores,
    SmStats,
    SmApply,
    AttnAv,
    Glu,
    DwConv,
    Conv0Dw1,
    DwS2,
    Flatten,
}

#[derive(Clone, Copy)]
enum Rows {
    T,
    Pos,
    Sub1,
    Sub2,
}

#[derive(Clone, Copy)]
enum Op {
    Gemm { rows: Rows, k: i32, n: i32, mode: i32, alpha: f32 },
    LnStats,
    LnApply,
    Scores,
    SmStats,
    SmApply,
    AttnAv,
    Glu,
    DwConv,
    Conv0Dw1,
    DwS2,
    Flatten,
}

struct Step {
    prog: P,
    inputs: Vec<Tex>,
    out: Tex,
    op: Op,
}

#[derive(Clone, Copy)]
struct Dims {
    frames: i32,
    h1: i32,
    h2: i32,
    h3: i32,
    t: i32,
    cap: i32,
}

impl Op {
    /// (ua, uf, output texels)
    fn params(&self, d: Dims) -> ([i32; 4], [f32; 4], i32) {
        let t = d.t;
        let t4 = (t + 3) / 4;
        let ctx = [t, d.cap, t4, 0];
        match *self {
            Op::Gemm { rows, k, n, mode, alpha } => {
                let m = match rows {
                    Rows::T => t,
                    Rows::Pos => 2 * d.cap - 1,
                    Rows::Sub1 => d.h2 * 32,
                    Rows::Sub2 => d.h3 * 16,
                };
                ([m, k, n, mode], [alpha, 0.0, 0.0, 0.0], m * n / 4)
            }
            Op::LnStats => (ctx, [0.0; 4], t),
            Op::LnApply | Op::Glu | Op::DwConv => (ctx, [0.0; 4], t * 256),
            Op::Scores => (ctx, [1.0 / (HEAD_DIM as f32).sqrt(), 0.0, 0.0, 0.0], N_HEADS as i32 * t * t4),
            Op::SmStats => (ctx, [0.0; 4], N_HEADS as i32 * t),
            Op::SmApply => (ctx, [0.0; 4], N_HEADS as i32 * t * t4),
            Op::AttnAv => (ctx, [0.0; 4], t * 256),
            Op::Conv0Dw1 => ([d.frames, d.h1, d.h2, 0], [0.0; 4], d.h2 * 32 * 64),
            Op::DwS2 => ([d.h2, 32, d.h3, 16], [0.0; 4], d.h3 * 16 * 64),
            Op::Flatten => (ctx, [0.0; 4], t * 1024),
        }
    }
}

struct Plan {
    cap: usize,
    mel: Tex,
    out: Tex,
    steps: Vec<Step>,
    /// every texture the plan allocated
    own: Vec<Tex>,
}

struct LayerW {
    mats: [(Tex, Tex); 11], // ff1_w1, ff1_w2, q, k, v, o, pos, pw1, pw2, ff2_w1, ff2_w2: (planes, lohi)
    bias_u: Tex,
    bias_v: Tex,
    dw: Tex,
    bn_scale: Tex,
    bn_shift: Tex,
    ln: [(Tex, Tex); 5],
}

struct Weights {
    c0_w: Tex,
    c0_b: Tex,
    dw1_w: Tex,
    dw1_b: Tex,
    pw1: (Tex, Tex),
    dw2_w: Tex,
    dw2_b: Tex,
    pw2: (Tex, Tex),
    linear: (Tex, Tex),
    proj: (Tex, Tex),
    layers: Vec<LayerW>,
    dummy: Tex,
}

pub struct GlEncoder {
    gl: Gl,
    _canvas: OffscreenCanvas,
    fbo: WebGlFramebuffer,
    progs: Vec<Prog>,
    w: Weights,
    plan: Option<Plan>,
    info: String,
}

fn err(e: JsValue) -> String {
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}

impl GlEncoder {
    pub fn new(model: &Model) -> Result<GlEncoder, String> {
        let canvas = OffscreenCanvas::new(1, 1).map_err(err)?;
        let attrs = Object::new();
        for (k, v) in [("antialias", false), ("depth", false), ("stencil", false), ("alpha", true), ("preserveDrawingBuffer", false)] {
            Reflect::set(&attrs, &k.into(), &v.into()).map_err(err)?;
        }
        Reflect::set(&attrs, &"powerPreference".into(), &"high-performance".into()).map_err(err)?;
        let gl: Gl = canvas
            .get_context_with_context_options("webgl2", &attrs)
            .map_err(err)?
            .ok_or("WebGL2 is not available")?
            .dyn_into()
            .map_err(|_| "not a WebGL2 context".to_string())?;
        if gl.get_extension("EXT_color_buffer_float").map_err(err)?.is_none() {
            return Err("WebGL2 lacks EXT_color_buffer_float (float render targets)".into());
        }
        let max_tex = gl.get_parameter(Gl::MAX_TEXTURE_SIZE).map_err(err)?.as_f64().unwrap_or(0.0) as i32;
        if max_tex < 4096 {
            return Err(format!("WebGL2 MAX_TEXTURE_SIZE {max_tex} < 4096"));
        }
        let renderer = renderer_name(&gl);
        let vs = compile(&gl, Gl::VERTEX_SHADER, shaders::VERTEX)?;
        let mut progs = Vec::new();
        for k in shaders::kernels() {
            let fs = compile(&gl, Gl::FRAGMENT_SHADER, &k.src).map_err(|e| format!("{}: {e}", k.name))?;
            let p = gl.create_program().ok_or("create_program")?;
            gl.attach_shader(&p, &vs);
            gl.attach_shader(&p, &fs);
            gl.link_program(&p);
            if !gl.get_program_parameter(&p, Gl::LINK_STATUS).as_bool().unwrap_or(false) {
                return Err(format!("{}: link: {}", k.name, gl.get_program_info_log(&p).unwrap_or_default()));
            }
            gl.use_program(Some(&p));
            for (unit, s) in k.samplers.iter().enumerate() {
                gl.uniform1i(gl.get_uniform_location(&p, s).as_ref(), unit as i32);
            }
            progs.push(Prog {
                ua: gl.get_uniform_location(&p, "ua"),
                ub: gl.get_uniform_location(&p, "ub"),
                uf: gl.get_uniform_location(&p, "uf"),
                nsamp: k.samplers.len(),
                p,
            });
        }
        let vao = gl.create_vertex_array().ok_or("vao")?;
        gl.bind_vertex_array(Some(&vao));
        let fbo = gl.create_framebuffer().ok_or("fbo")?;
        gl.disable(Gl::DITHER);
        gl.disable(Gl::BLEND);
        let w = upload(&gl, model)?;
        Ok(GlEncoder { gl, _canvas: canvas, fbo, progs, w, plan: None, info: renderer })
    }

    pub fn adapter_info(&self) -> String {
        self.info.clone()
    }

    /// Projected encoder output `[T][640]` for normalised log-mel features `[frames][128]`.
    pub fn encode(&mut self, mel: &[f32], frames: usize) -> Result<(Vec<f32>, usize), String> {
        let t = encoder_frames(frames);
        if t == 0 {
            return Ok((Vec::new(), 0));
        }
        if t > MAX_FRAMES {
            return Err(format!("window of {t} frames exceeds {MAX_FRAMES}"));
        }
        if self.plan.as_ref().is_none_or(|p| p.cap < t) {
            if let Some(old) = self.plan.take() {
                self.free_plan(old);
            }
            let cap = t.div_ceil(64) * 64;
            self.plan = Some(self.build_plan(cap)?);
        }
        let plan = self.plan.as_ref().unwrap();
        upload_f32_into(&self.gl, &plan.mel, mel)?;
        let d = dims(frames, plan.cap);
        for s in &plan.steps {
            self.draw(s, d);
        }
        let n = t * PRED;
        let out = read_tex(&self.gl, &self.fbo, &plan.out, n)?;
        Ok((out, t))
    }

    fn draw(&self, s: &Step, d: Dims) {
        let gl = &self.gl;
        let pr = &self.progs[s.prog as usize];
        let (ua, uf, texels) = s.op.params(d);
        if texels <= 0 {
            return;
        }
        gl.use_program(Some(&pr.p));
        gl.uniform4i(pr.ua.as_ref(), ua[0], ua[1], ua[2], ua[3]);
        gl.uniform4i(pr.ub.as_ref(), 0, 0, 0, texels);
        gl.uniform4f(pr.uf.as_ref(), uf[0], uf[1], uf[2], uf[3]);
        debug_assert_eq!(pr.nsamp, s.inputs.len());
        for (u, tex) in s.inputs.iter().enumerate() {
            gl.active_texture(Gl::TEXTURE0 + u as u32);
            gl.bind_texture(Gl::TEXTURE_2D, Some(&tex.t));
        }
        gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(&self.fbo));
        gl.framebuffer_texture_2d(Gl::FRAMEBUFFER, Gl::COLOR_ATTACHMENT0, Gl::TEXTURE_2D, Some(&s.out.t), 0);
        let rows = (texels as usize).div_ceil(WIDTH as usize) as i32;
        gl.viewport(0, 0, WIDTH, rows);
        gl.draw_arrays(Gl::TRIANGLES, 0, 3);
        // unbind inputs so the next draw may render into one of them
        for u in 0..s.inputs.len() {
            gl.active_texture(Gl::TEXTURE0 + u as u32);
            gl.bind_texture(Gl::TEXTURE_2D, None);
        }
    }

    fn free_plan(&self, p: Plan) {
        for t in p.own {
            self.gl.delete_texture(Some(&t.t));
        }
    }

    fn build_plan(&self, cap: usize) -> Result<Plan, String> {
        let gl = &self.gl;
        let w = &self.w;
        let f = |texels: usize| -> Result<Tex, String> { alloc(gl, Gl::RGBA32F, texels) };
        let t4 = cap.div_ceil(4);
        let mel = f(8 * cap * N_MELS / 4)?;
        let s1 = f(2 * cap * 32 * SUB_CH / 4)?;
        let s2 = f(2 * cap * 32 * SUB_CH / 4)?;
        let xa = f(cap * D_MODEL / 4)?;
        let xb = f(cap * D_MODEL / 4)?;
        let h = f(cap * D_MODEL / 4)?;
        let ff = f(cap * D_FF / 4)?;
        let qb = f(cap * D_MODEL / 4)?;
        let kb = f(cap * D_MODEL / 4)?;
        let vb = f(cap * D_MODEL / 4)?;
        let ctx = f(cap * D_MODEL / 4)?;
        let st = f(N_HEADS * cap)?;
        let sc = f(N_HEADS * cap * t4)?;
        let pr = f(N_HEADS * cap * t4)?;
        let out = f(cap * PRED / 4)?;
        let mut own = vec![mel.clone(), s1.clone(), s2.clone(), xa.clone(), xb.clone(), h.clone(), ff.clone(), qb.clone(), kb.clone(), vb.clone(), ctx.clone(), st.clone(), sc.clone(), pr.clone(), out.clone()];
        let pe = alloc(gl, Gl::RGBA32F, (2 * cap - 1) * D_MODEL / 4)?;
        upload_f32_into(gl, &pe, &pos_embedding(cap))?;
        let d = w.dummy.clone();

        // position projections, run now
        let mut pos = Vec::new();
        for lw in &w.layers {
            let p = f((2 * cap - 1) * D_MODEL / 4)?;
            let (pl, lh) = &lw.mats[6];
            let s = Step { prog: P::GemmFv, inputs: vec![pe.clone(), pl.clone(), lh.clone(), d.clone(), d.clone()], out: p.clone(), op: gemm(Rows::Pos, D_MODEL, D_MODEL, 0, 0.0) };
            self.draw(&s, dims(8 * cap, cap));
            own.push(p.clone());
            pos.push(p);
        }
        gl.delete_texture(Some(&pe.t));

        let mut steps = Vec::new();
        let fv = |x: &Tex, m: &(Tex, Tex), y: &Tex, r: &Tex, rows: usize, cols: usize, mode: i32, alpha: f32| Step {
            prog: P::GemmFv,
            inputs: vec![x.clone(), m.0.clone(), m.1.clone(), d.clone(), r.clone()],
            out: y.clone(),
            op: gemm(Rows::T, cols, rows, mode, alpha),
        };
        let dense = |x: &Tex, m: &(Tex, Tex), y: &Tex, rows_kind: Rows, n: usize, k: usize, mode: i32| Step {
            prog: P::GemmDense,
            inputs: vec![x.clone(), m.0.clone(), m.1.clone(), d.clone()],
            out: y.clone(),
            op: gemm(rows_kind, k, n, mode, 0.0),
        };
        // subsampling
        steps.push(Step { prog: P::Conv0Dw1, inputs: vec![mel.clone(), w.c0_w.clone(), w.c0_b.clone(), w.dw1_w.clone(), w.dw1_b.clone()], out: s1.clone(), op: Op::Conv0Dw1 });
        steps.push(dense(&s1, &w.pw1, &s2, Rows::Sub1, SUB_CH, SUB_CH, 4));
        steps.push(Step { prog: P::DwS2, inputs: vec![s2.clone(), w.dw2_w.clone(), w.dw2_b.clone()], out: s1.clone(), op: Op::DwS2 });
        steps.push(dense(&s1, &w.pw2, &s2, Rows::Sub2, SUB_CH, SUB_CH, 4));
        steps.push(Step { prog: P::Flatten, inputs: vec![s2.clone()], out: ff.clone(), op: Op::Flatten });
        steps.push(dense(&ff, &w.linear, &xa, Rows::T, D_MODEL, D_FF, 1));
        // the residual stream ping-pongs between xa and xb
        let mut cur = xa.clone();
        let mut other = xb.clone();
        for (l, lw) in w.layers.iter().enumerate() {
            let ln = |steps: &mut Vec<Step>, src: &Tex, dst: &Tex, k: usize| {
                steps.push(Step { prog: P::LnStats, inputs: vec![src.clone()], out: st.clone(), op: Op::LnStats });
                steps.push(Step { prog: P::LnApply, inputs: vec![src.clone(), st.clone(), lw.ln[k].0.clone(), lw.ln[k].1.clone()], out: dst.clone(), op: Op::LnApply });
            };
            let m = &lw.mats;
            // ff1
            ln(&mut steps, &cur, &h, 0);
            steps.push(fv(&h, &m[0], &ff, &d, D_FF, D_MODEL, 3, 0.0));
            steps.push(fv(&ff, &m[1], &other, &cur, D_MODEL, D_FF, 2, 0.5));
            std::mem::swap(&mut cur, &mut other);
            // attention
            ln(&mut steps, &cur, &h, 1);
            steps.push(fv(&h, &m[2], &qb, &d, D_MODEL, D_MODEL, 0, 0.0));
            steps.push(fv(&h, &m[3], &kb, &d, D_MODEL, D_MODEL, 0, 0.0));
            steps.push(fv(&h, &m[4], &vb, &d, D_MODEL, D_MODEL, 0, 0.0));
            steps.push(Step { prog: P::Scores, inputs: vec![qb.clone(), kb.clone(), pos[l].clone(), lw.bias_u.clone(), lw.bias_v.clone()], out: sc.clone(), op: Op::Scores });
            steps.push(Step { prog: P::SmStats, inputs: vec![sc.clone()], out: st.clone(), op: Op::SmStats });
            steps.push(Step { prog: P::SmApply, inputs: vec![sc.clone(), st.clone()], out: pr.clone(), op: Op::SmApply });
            steps.push(Step { prog: P::AttnAv, inputs: vec![pr.clone(), vb.clone()], out: ctx.clone(), op: Op::AttnAv });
            steps.push(fv(&ctx, &m[5], &other, &cur, D_MODEL, D_MODEL, 2, 1.0));
            std::mem::swap(&mut cur, &mut other);
            // conv module
            ln(&mut steps, &cur, &h, 2);
            steps.push(fv(&h, &m[7], &ff, &d, 2 * D_MODEL, D_MODEL, 0, 0.0));
            steps.push(Step { prog: P::Glu, inputs: vec![ff.clone()], out: qb.clone(), op: Op::Glu });
            steps.push(Step { prog: P::DwConv, inputs: vec![qb.clone(), lw.dw.clone(), lw.bn_scale.clone(), lw.bn_shift.clone()], out: kb.clone(), op: Op::DwConv });
            steps.push(fv(&kb, &m[8], &other, &cur, D_MODEL, D_MODEL, 2, 1.0));
            std::mem::swap(&mut cur, &mut other);
            // ff2
            ln(&mut steps, &cur, &h, 3);
            steps.push(fv(&h, &m[9], &ff, &d, D_FF, D_MODEL, 3, 0.0));
            steps.push(fv(&ff, &m[10], &other, &cur, D_MODEL, D_FF, 2, 0.5));
            std::mem::swap(&mut cur, &mut other);
            // ln_out
            ln(&mut steps, &cur, &other, 4);
            std::mem::swap(&mut cur, &mut other);
        }
        steps.push(dense(&cur, &w.proj, &out, Rows::T, PRED, D_MODEL, 1));
        Ok(Plan { cap, mel, out, steps, own })
    }
}

impl Drop for GlEncoder {
    /// Free the GPU memory now instead of whenever the JS garbage collector gets to the textures.
    fn drop(&mut self) {
        if let Ok(Some(ext)) = self.gl.get_extension("WEBGL_lose_context") {
            let _ = js_sys::Reflect::get(&ext, &"loseContext".into())
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .map(|f| f.call0(&ext));
        }
    }
}

fn gemm(rows: Rows, k: usize, n: usize, mode: i32, alpha: f32) -> Op {
    Op::Gemm { rows, k: k as i32, n: n as i32, mode, alpha }
}

fn dims(frames: usize, cap: usize) -> Dims {
    let h1 = frames.div_ceil(2);
    let h2 = h1.div_ceil(2);
    let h3 = h2.div_ceil(2);
    Dims { frames: frames as i32, h1: h1 as i32, h2: h2 as i32, h3: h3 as i32, t: h3 as i32, cap: cap as i32 }
}

fn renderer_name(gl: &Gl) -> String {
    let raw = gl
        .get_extension("WEBGL_debug_renderer_info")
        .ok()
        .flatten()
        .and_then(|_| gl.get_parameter(0x9246).ok())
        .and_then(|v| v.as_string())
        .or_else(|| gl.get_parameter(Gl::RENDERER).ok().and_then(|v| v.as_string()))
        .unwrap_or_default();
    tidy_renderer(&raw)
}

/// "ANGLE (Apple, ANGLE Metal Renderer: Apple M5 Pro, Unspecified Version)" -> "Apple M5 Pro";
/// "ANGLE (Intel, Intel(R) Iris(TM) Graphics 6100 Direct3D11 vs_5_0 ps_5_0, D3D11)" -> "Intel(R) Iris(TM) Graphics 6100".
fn tidy_renderer(raw: &str) -> String {
    let mut s = raw.trim();
    if let Some(inner) = s.strip_prefix("ANGLE (").and_then(|x| x.strip_suffix(')')) {
        s = inner.split(", ").nth(1).unwrap_or(inner);
    }
    if let Some(i) = s.find("Renderer: ") {
        s = &s[i + 10..];
    }
    for cut in [" Direct3D", " (0x", ", D3D", " OpenGL", " Vulkan"] {
        if let Some(i) = s.find(cut) {
            s = &s[..i];
        }
    }
    let s = s.trim();
    if s.is_empty() { "GPU".to_string() } else { s.to_string() }
}

fn compile(gl: &Gl, kind: u32, src: &str) -> Result<web_sys::WebGlShader, String> {
    let s = gl.create_shader(kind).ok_or("create_shader")?;
    gl.shader_source(&s, src);
    gl.compile_shader(&s);
    if !gl.get_shader_parameter(&s, Gl::COMPILE_STATUS).as_bool().unwrap_or(false) {
        return Err(gl.get_shader_info_log(&s).unwrap_or_default());
    }
    Ok(s)
}

/// An immutable 1024-wide texture of `texels` RGBA texels.
fn alloc(gl: &Gl, internal: u32, texels: usize) -> Result<Tex, String> {
    let rows = texels.max(1).div_ceil(WIDTH as usize) as i32;
    let t = gl.create_texture().ok_or("create_texture")?;
    gl.bind_texture(Gl::TEXTURE_2D, Some(&t));
    gl.tex_storage_2d(Gl::TEXTURE_2D, 1, internal, WIDTH, rows);
    gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::NEAREST as i32);
    gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::NEAREST as i32);
    gl.bind_texture(Gl::TEXTURE_2D, None);
    let e = gl.get_error();
    if e != Gl::NO_ERROR {
        return Err(format!("texture of {texels} texels: GL error {e:#x} (out of GPU memory?)"));
    }
    Ok(Tex { t })
}

/// Upload floats (padded to whole rows) into a texture. The data is copied into a fresh JS array first: WebGL may
/// refuse views of the shared wasm memory.
fn upload_f32_into(gl: &Gl, tex: &Tex, data: &[f32]) -> Result<(), String> {
    let texels = data.len().div_ceil(4);
    let rows = texels.div_ceil(WIDTH as usize);
    let arr = Float32Array::new_with_length((rows * WIDTH as usize * 4) as u32);
    arr.subarray(0, data.len() as u32).copy_from(data);
    gl.bind_texture(Gl::TEXTURE_2D, Some(&tex.t));
    gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(Gl::TEXTURE_2D, 0, 0, 0, WIDTH, rows as i32, Gl::RGBA, Gl::FLOAT, Some(&arr))
        .map_err(err)?;
    gl.bind_texture(Gl::TEXTURE_2D, None);
    Ok(())
}

fn tex_f32(gl: &Gl, data: &[f32]) -> Result<Tex, String> {
    let t = alloc(gl, Gl::RGBA32F, data.len().div_ceil(4))?;
    upload_f32_into(gl, &t, data)?;
    Ok(t)
}

fn tex_u32(gl: &Gl, data: &[u32]) -> Result<Tex, String> {
    let texels = data.len().div_ceil(4);
    let t = alloc(gl, Gl::RGBA32UI, texels)?;
    let rows = texels.div_ceil(WIDTH as usize);
    let arr = Uint32Array::new_with_length((rows * WIDTH as usize * 4) as u32);
    arr.subarray(0, data.len() as u32).copy_from(data);
    gl.bind_texture(Gl::TEXTURE_2D, Some(&t.t));
    gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(Gl::TEXTURE_2D, 0, 0, 0, WIDTH, rows as i32, Gl::RGBA_INTEGER, Gl::UNSIGNED_INT, Some(&arr))
        .map_err(err)?;
    gl.bind_texture(Gl::TEXTURE_2D, None);
    Ok(t)
}

fn read_tex(gl: &Gl, fbo: &WebGlFramebuffer, tex: &Tex, floats: usize) -> Result<Vec<f32>, String> {
    let rows = floats.div_ceil(4).div_ceil(WIDTH as usize);
    gl.bind_framebuffer(Gl::FRAMEBUFFER, Some(fbo));
    gl.framebuffer_texture_2d(Gl::FRAMEBUFFER, Gl::COLOR_ATTACHMENT0, Gl::TEXTURE_2D, Some(&tex.t), 0);
    let arr = Float32Array::new_with_length((rows * WIDTH as usize * 4) as u32);
    gl.read_pixels_with_opt_array_buffer_view(0, 0, WIDTH, rows as i32, Gl::RGBA, Gl::FLOAT, Some(&arr)).map_err(err)?;
    let e = gl.get_error();
    if e != Gl::NO_ERROR {
        return Err(format!("readPixels: GL error {e:#x}"));
    }
    let mut v = vec![0f32; floats];
    arr.subarray(0, floats as u32).copy_to(&mut v);
    Ok(v)
}

/// Two 2-bit ternary planes per row: per 32 columns one uvec4 (lo k0..15, hi k0..15, lo k16..31, hi k16..31), each
/// field `t + 1` with `t` the sign of a weight in that plane (0 when the weight is in the other plane or zero).
fn planes(m: &FvMatrix) -> Vec<u32> {
    assert_eq!(m.cols % 32, 0);
    // one code byte (two weights) -> 4 bits of the lo plane and 4 bits of the hi plane
    const LUT: [(u8, u8); 256] = {
        let mut t = [(0u8, 0u8); 256];
        let mut b = 0;
        while b < 256 {
            let mut lo = 0u8;
            let mut hi = 0u8;
            let mut k = 0;
            while k < 2 {
                let c = ((b >> (4 * k)) & 15) as u8;
                let (tl, th): (u8, u8) = match c {
                    CODE_POS_LO => (2, 1),
                    CODE_NEG_LO => (0, 1),
                    CODE_POS_HI => (1, 2),
                    CODE_NEG_HI => (1, 0),
                    _ => (1, 1),
                };
                lo |= tl << (2 * k);
                hi |= th << (2 * k);
                k += 1;
            }
            t[b] = (lo, hi);
            b += 1;
        }
        t
    };
    let kw = m.cols / 32;
    let half = m.cols / 2;
    let mut out = vec![0u32; m.rows * kw * 4];
    for r in 0..m.rows {
        let row = &m.codes[r * half..(r + 1) * half];
        for kb in 0..kw {
            let bytes = &row[kb * 16..kb * 16 + 16];
            let mut words = [0u32; 4];
            for (i, &byte) in bytes.iter().enumerate() {
                let (lo, hi) = LUT[byte as usize];
                let (wl, sh) = if i < 8 { (0, 4 * i) } else { (2, 4 * (i - 8)) };
                words[wl] |= (lo as u32) << sh;
                words[wl + 1] |= (hi as u32) << sh;
            }
            out[(r * kw + kb) * 4..(r * kw + kb) * 4 + 4].copy_from_slice(&words);
        }
    }
    out
}

/// lo/hi per 4-row block: texel 2b = lo[4b..4b+4], texel 2b+1 = hi[4b..4b+4].
fn lohi(m: &FvMatrix) -> Vec<f32> {
    let mut v = Vec::with_capacity(2 * m.rows);
    for b in 0..m.rows / 4 {
        v.extend_from_slice(&m.lo[4 * b..4 * b + 4]);
        v.extend_from_slice(&m.hi[4 * b..4 * b + 4]);
    }
    v
}

/// `[C][taps]` -> `[taps][C]`
fn tap_major(w: &[f32], taps: usize) -> Vec<f32> {
    let c = w.len() / taps;
    let mut t = vec![0f32; w.len()];
    for ch in 0..c {
        for k in 0..taps {
            t[k * c + ch] = w[ch * taps + k];
        }
    }
    t
}

fn upload(gl: &Gl, m: &Model) -> Result<Weights, String> {
    let dense = |q: &QMatrix, b: &[f32]| -> Result<(Tex, Tex), String> { Ok((tex_f32(gl, &q.dense())?, tex_f32(gl, b)?)) };
    let s = &m.sub;
    let mut layers = Vec::new();
    for l in &m.layers {
        let mats = [&l.ff1_w1, &l.ff1_w2, &l.q, &l.k, &l.v, &l.o, &l.pos, &l.pw1, &l.pw2, &l.ff2_w1, &l.ff2_w2];
        // pack the layer's planes on the thread pool, upload on this (the GL) thread
        let packed = phonon_core::par::map(mats.len(), |i| planes(mats[i]));
        let mut v = Vec::new();
        for (x, p) in mats.iter().zip(&packed) {
            v.push((tex_u32(gl, p)?, tex_f32(gl, &lohi(x))?));
        }
        let mut ln = Vec::new();
        for n in [&l.ln_ff1, &l.ln_att, &l.ln_conv, &l.ln_ff2, &l.ln_out] {
            ln.push((tex_f32(gl, &n.w)?, tex_f32(gl, &n.b)?));
        }
        layers.push(LayerW {
            mats: v.try_into().map_err(|_| "mats")?,
            bias_u: tex_f32(gl, &l.bias_u)?,
            bias_v: tex_f32(gl, &l.bias_v)?,
            dw: tex_f32(gl, &tap_major(&l.dw, 9))?,
            bn_scale: tex_f32(gl, &l.bn_scale)?,
            bn_shift: tex_f32(gl, &l.bn_shift)?,
            ln: ln.try_into().map_err(|_| "ln")?,
        });
    }
    Ok(Weights {
        c0_w: tex_f32(gl, &tap_major(&s.c0_w, 9))?,
        c0_b: tex_f32(gl, &s.c0_b)?,
        dw1_w: tex_f32(gl, &tap_major(&s.dw1_w, 9))?,
        dw1_b: tex_f32(gl, &s.dw1_b)?,
        pw1: dense(&s.pw1, &s.pw1_b)?,
        dw2_w: tex_f32(gl, &tap_major(&s.dw2_w, 9))?,
        dw2_b: tex_f32(gl, &s.dw2_b)?,
        pw2: dense(&s.pw2, &s.pw2_b)?,
        linear: dense(&s.linear, &s.linear_b)?,
        proj: dense(&m.enc_proj, &m.enc_proj_b)?,
        layers,
        dummy: tex_f32(gl, &[0.0; 4])?,
    })
}
