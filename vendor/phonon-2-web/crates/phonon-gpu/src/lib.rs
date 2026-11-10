//! The Phonon-2 encoder on the GPU through wgpu: WebGPU in the browser, Metal / Vulkan / D3D12 natively.
//!
//! Weights stay in their compact form on the GPU (the five-value matrices as 4-bit codes plus per-row lo/hi, ~290 MB)
//! and are expanded inside the GEMM tiles. Activations are f32, so results match the CPU path to float rounding.
//!
//! The whole pass (subsampling, 24 conformer layers, projector; ~490 dispatches) is recorded against buffers sized for
//! a capacity of `cap` encoder frames. Bind groups are built once per capacity; each call only writes one uniform buffer
//! of per-step parameters (which depend on the clip length), submits once and reads back the `[T][640]` projected
//! output for the CPU decoder.

mod shaders;

use phonon_core::encoder::{encoder_frames, pos_embedding};
use phonon_core::model::{Model, D_FF, D_MODEL, HEAD_DIM, N_HEADS, N_MELS, PRED, SUB_CH};
use phonon_core::tensor::{FvMatrix, QMatrix};
use wgpu::util::DeviceExt;

/// Longest window the plan accepts, in encoder frames (~41 s; windows are at most 35 s).
pub const MAX_FRAMES: usize = 512;
const SLOT: u64 = 256;

/// Receives a stage name as initialisation proceeds.
pub type Log = Box<dyn Fn(&str)>;

#[derive(Default)]
pub struct Options {
    /// Prefer an integrated GPU (saves battery); default is the high-performance adapter.
    pub low_power: bool,
    /// Called with a stage name as initialisation proceeds (for timing).
    pub log: Option<Log>,
}

#[derive(Debug)]
pub struct GpuError(pub String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

type Result<T> = std::result::Result<T, GpuError>;

#[derive(Clone, Copy)]
#[repr(usize)]
enum K {
    GemmFv,
    GemmDense,
    Ln,
    LnInplace,
    Scores,
    Softmax,
    AttnAv,
    Glu,
    DwConv,
    Conv0Dw1,
    DwS2,
}

struct Fv {
    codes: wgpu::Buffer,
    lut: wgpu::Buffer,
    rows: u32,
    cols: u32,
}

struct Dense {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
    rows: u32,
    cols: u32,
}

struct LayerW {
    ff1_w1: Fv,
    ff1_w2: Fv,
    q: Fv,
    k: Fv,
    v: Fv,
    o: Fv,
    pos: Fv,
    pw1: Fv,
    pw2: Fv,
    ff2_w1: Fv,
    ff2_w2: Fv,
    bias_u: wgpu::Buffer,
    bias_v: wgpu::Buffer,
    dw: wgpu::Buffer,
    bn_scale: wgpu::Buffer,
    bn_shift: wgpu::Buffer,
    /// (gamma, beta) for ff1, att, conv, ff2, out
    ln: [(wgpu::Buffer, wgpu::Buffer); 5],
}

struct Weights {
    c0_w: wgpu::Buffer,
    c0_b: wgpu::Buffer,
    dw1_w: wgpu::Buffer,
    dw1_b: wgpu::Buffer,
    pw1: Dense,
    dw2_w: wgpu::Buffer,
    dw2_b: wgpu::Buffer,
    pw2: Dense,
    linear: Dense,
    proj: Dense,
    layers: Vec<LayerW>,
    zeros: wgpu::Buffer,
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
    Gemm { rows: Rows, k: u32, n: u32, mode: u32, alpha: f32 },
    Ln,
    Scores,
    Softmax,
    AttnAv,
    Glu,
    DwConv,
    Conv0Dw1,
    DwS2,
}

struct Step {
    kernel: K,
    bg: wgpu::BindGroup,
    op: Op,
}

#[derive(Clone, Copy)]
struct Dims {
    frames: u32,
    h1: u32,
    h2: u32,
    h3: u32,
    t: u32,
    cap: u32,
}

impl Dims {
    fn new(frames: usize, cap: usize) -> Dims {
        let h1 = frames.div_ceil(2);
        let h2 = h1.div_ceil(2);
        let h3 = h2.div_ceil(2);
        Dims { frames: frames as u32, h1: h1 as u32, h2: h2 as u32, h3: h3 as u32, t: h3 as u32, cap: cap as u32 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
    f: u32,
    g: f32,
    h: f32,
}

fn groups_1d(n: u32) -> [u32; 3] {
    let g = n.div_ceil(256);
    [g.min(65535), g.div_ceil(65535), 1]
}

impl Op {
    fn launch(&self, d: Dims) -> (Params, [u32; 3]) {
        let t = d.t;
        let mut p = Params::default();
        let g = match *self {
            Op::Gemm { rows, k, n, mode, alpha } => {
                let m = match rows {
                    Rows::T => t,
                    Rows::Pos => 2 * d.cap - 1,
                    Rows::Sub1 => d.h2 * 32,
                    Rows::Sub2 => d.h3 * 16,
                };
                p = Params { a: m, b: k, c: n, d: mode, g: alpha, ..p };
                [n / 64, m.div_ceil(64), 1]
            }
            Op::Ln => {
                p.a = t;
                [t.min(65535), t.div_ceil(65535), 1]
            }
            Op::Scores => {
                p = Params { a: t, b: d.cap, g: 1.0 / (HEAD_DIM as f32).sqrt(), ..p };
                [t.div_ceil(16), t.div_ceil(16), N_HEADS as u32]
            }
            Op::Softmax => {
                p.a = t;
                [t, N_HEADS as u32, 1]
            }
            Op::AttnAv => {
                p.a = t;
                [(HEAD_DIM / 64) as u32, t.div_ceil(64), N_HEADS as u32]
            }
            Op::Glu => {
                p.a = t * D_MODEL as u32;
                groups_1d(p.a)
            }
            Op::DwConv => {
                p.a = t;
                groups_1d(t * D_MODEL as u32)
            }
            Op::Conv0Dw1 => {
                p = Params { a: d.frames, b: d.h1, c: d.h2, ..p };
                groups_1d(d.h2 * 32 * SUB_CH as u32)
            }
            Op::DwS2 => {
                p = Params { a: d.h2, b: 32, c: d.h3, d: 16, ..p };
                groups_1d(d.h3 * 16 * SUB_CH as u32)
            }
        };
        (p, g)
    }
}

/// Buffers and recorded steps for clips of up to `cap` encoder frames.
struct Plan {
    cap: usize,
    mel: wgpu::Buffer,
    out: wgpu::Buffer,
    staging: wgpu::Buffer,
    params: wgpu::Buffer,
    steps: Vec<Step>,
    _keep: Vec<wgpu::Buffer>,
}

pub struct GpuEncoder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: String,
    pipes: Vec<wgpu::ComputePipeline>,
    w: Weights,
    plan: Option<Plan>,
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

impl GpuEncoder {
    pub async fn new(model: &Model, opts: Options) -> Result<GpuEncoder> {
        let log = |m: &str| {
            if let Some(f) = &opts.log {
                f(m)
            }
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if opts.low_power { wgpu::PowerPreference::LowPower } else { wgpu::PowerPreference::HighPerformance },
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError(format!("no GPU adapter: {e}")))?;
        let ai = adapter.get_info();
        let al = adapter.limits();
        let need = 64 << 20;
        if (al.max_storage_buffer_binding_size as u64) < need {
            return Err(GpuError(format!("adapter allows {} MB storage bindings, need 64 MB", al.max_storage_buffer_binding_size >> 20)));
        }
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size.min(256 << 20);
        limits.max_buffer_size = al.max_buffer_size.min(256 << 20);
        limits.max_storage_buffers_per_shader_stage = 8;
        limits.max_compute_workgroup_storage_size = 16384;
        limits.max_compute_invocations_per_workgroup = 256;
        limits.max_compute_workgroup_size_x = 256;
        limits.max_compute_workgroup_size_y = 256;
        limits.max_compute_workgroups_per_dimension = 65535;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("phonon"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError(format!("request_device: {e}")))?;
        // browsers expose little or nothing about the adapter
        let info = match (ai.name.trim(), ai.backend) {
            ("", wgpu::Backend::BrowserWebGpu) => "GPU".to_string(),
            (n, wgpu::Backend::BrowserWebGpu) => n.to_string(),
            ("", b) => format!("GPU {:04x}:{:04x} ({b:?})", ai.vendor, ai.device),
            (n, b) => format!("{n} ({b:?})"),
        };
        log("device");

        let srcs = [
            shaders::gemm_fv(),
            shaders::gemm_dense(),
            shaders::layernorm(false),
            shaders::layernorm(true),
            shaders::scores(),
            shaders::softmax(),
            shaders::attn_av(),
            shaders::glu(),
            shaders::dwconv(),
            shaders::conv0_dw1(),
            shaders::dw_s2(),
        ];
        let pipes = srcs
            .iter()
            .enumerate()
            .map(|(i, src)| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("k{i}")),
                    source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("k{i}")),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        log("pipelines");
        let w = upload(&device, model);
        log("weights");
        Ok(GpuEncoder { device, queue, info, pipes, w, plan: None })
    }

    pub fn adapter_info(&self) -> String {
        self.info.clone()
    }

    /// Projected encoder output `[T][640]` for normalised log-mel features `[frames][128]`.
    pub async fn encode(&mut self, mel: &[f32], frames: usize) -> Result<(Vec<f32>, usize)> {
        assert_eq!(mel.len(), frames * N_MELS);
        let t = encoder_frames(frames);
        if t == 0 {
            return Ok((Vec::new(), 0));
        }
        if t > MAX_FRAMES {
            return Err(GpuError(format!("window of {t} frames exceeds the GPU plan's {MAX_FRAMES}")));
        }
        if self.plan.as_ref().is_none_or(|p| p.cap < t) {
            let cap = t.div_ceil(64) * 64;
            self.plan = None; // free the old buffers first
            self.plan = Some(self.build_plan(cap).await?);
        }
        let plan = self.plan.as_ref().unwrap();
        let d = Dims::new(frames, plan.cap);
        self.queue.write_buffer(&plan.mel, 0, f32_bytes(mel));
        let mut params = vec![0u8; plan.steps.len() * SLOT as usize];
        let mut launches = Vec::with_capacity(plan.steps.len());
        for (i, s) in plan.steps.iter().enumerate() {
            let (p, g) = s.op.launch(d);
            params[i * SLOT as usize..i * SLOT as usize + 32].copy_from_slice(bytemuck::bytes_of(&p));
            launches.push(g);
        }
        self.queue.write_buffer(&plan.params, 0, &params);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encode") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("encoder"), timestamp_writes: None });
            for (s, g) in plan.steps.iter().zip(&launches) {
                pass.set_pipeline(&self.pipes[s.kernel as usize]);
                pass.set_bind_group(0, &s.bg, &[]);
                pass.dispatch_workgroups(g[0], g[1], g[2]);
            }
        }
        let bytes = (t * PRED * 4) as u64;
        enc.copy_buffer_to_buffer(&plan.out, 0, &plan.staging, 0, bytes);
        self.queue.submit([enc.finish()]);
        let out = read_back(&self.device, &plan.staging, bytes).await?;
        Ok((out, t))
    }

    /// Native only: run the pass one step per submission and return wall time per kernel kind (for tuning).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn profile(&mut self, mel: &[f32], frames: usize) -> Vec<(String, f64, usize)> {
        let _ = pollster_block(self.encode(mel, frames));
        let plan = self.plan.as_ref().unwrap();
        let d = Dims::new(frames, plan.cap);
        let mut table: std::collections::BTreeMap<String, (f64, usize)> = Default::default();
        for s in &plan.steps {
            let (_, g) = s.op.launch(d);
            let name = match s.op {
                Op::Gemm { k, n, .. } => format!("{:?} {n}x{k}", s.kernel as usize),
                _ => format!("k{}", s.kernel as usize),
            };
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                let mut enc = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pipes[s.kernel as usize]);
                    pass.set_bind_group(0, &s.bg, &[]);
                    pass.dispatch_workgroups(g[0], g[1], g[2]);
                }
                self.queue.submit([enc.finish()]);
                poll_wait(&self.device);
            }
            let e = table.entry(name).or_default();
            e.0 += t0.elapsed().as_secs_f64() / 3.0;
            e.1 += 1;
        }
        let mut v: Vec<_> = table.into_iter().map(|(k, (t, n))| (k, t, n)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v
    }

    async fn build_plan(&self, cap: usize) -> Result<Plan> {
        let dev = &self.device;
        let w = &self.w;
        let mk = |label: &str, floats: usize| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (floats.max(4) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let npos = 2 * cap - 1;
        let mel = mk("mel", 8 * cap * N_MELS);
        let s1 = mk("sub1", 2 * cap * 32 * SUB_CH);
        let s2 = mk("sub2", 2 * cap * 32 * SUB_CH);
        let x = mk("x", cap * D_MODEL);
        let h = mk("h", cap * D_MODEL);
        let ff = mk("ff", cap * D_FF);
        let qb = mk("q", cap * D_MODEL);
        let kb = mk("k", cap * D_MODEL);
        let vb = mk("v", cap * D_MODEL);
        let ctx = mk("ctx", cap * D_MODEL);
        let sc = mk("scores", N_HEADS * cap * cap);
        let out = mk("out", cap * PRED);
        let pe = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pos_emb"),
            contents: f32_bytes(&pos_embedding(cap)),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pos: Vec<wgpu::Buffer> = (0..w.layers.len()).map(|l| mk(&format!("pos{l}"), npos * D_MODEL)).collect();
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (cap * PRED * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- the position projections (once per capacity; queue order puts them before any encode)
        let n_pos_steps = w.layers.len();
        let pos_params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pos_params"),
            size: n_pos_steps as u64 * SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut pos_steps = Vec::new();
        for (l, lw) in w.layers.iter().enumerate() {
            pos_steps.push(self.gemm_fv_step(&pos_params, l, &pe, &pos[l], &lw.pos, Rows::Pos, 0, 0.0));
        }
        self.run_once(&pos_params, &pos_steps, Dims::new(8 * cap, cap));

        // ---- the encoder pass
        // Count steps first so the params buffer can be created before the bind groups that reference it.
        let n_steps = 5 + 20 * w.layers.len() + 1;
        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: n_steps as u64 * SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut steps: Vec<Step> = Vec::with_capacity(n_steps);
        let mut slot = 0usize;
        let mut next = || {
            slot += 1;
            slot - 1
        };
        // subsampling
        steps.push(self.step(K::Conv0Dw1, &params, next(), &[(1, &mel), (2, &s1), (3, &w.c0_w), (4, &w.c0_b), (5, &w.dw1_w), (6, &w.dw1_b)], Op::Conv0Dw1));
        steps.push(self.gemm_dense_step(&params, next(), &s1, &s2, &w.pw1, Rows::Sub1, 4));
        steps.push(self.step(K::DwS2, &params, next(), &[(1, &s2), (2, &s1), (3, &w.dw2_w), (4, &w.dw2_b)], Op::DwS2));
        steps.push(self.gemm_dense_step(&params, next(), &s1, &ff, &w.pw2, Rows::Sub2, 5));
        steps.push(self.gemm_dense_step(&params, next(), &ff, &x, &w.linear, Rows::T, 1));
        for (l, lw) in w.layers.iter().enumerate() {
            let ln = |k: usize, src: &wgpu::Buffer, s: usize| {
                self.step(K::Ln, &params, s, &[(1, src), (2, &h), (3, &lw.ln[k].0), (4, &lw.ln[k].1)], Op::Ln)
            };
            // x += 0.5 * ff1(ln(x))
            steps.push(ln(0, &x, next()));
            steps.push(self.gemm_fv_step(&params, next(), &h, &ff, &lw.ff1_w1, Rows::T, 3, 0.0));
            steps.push(self.gemm_fv_step(&params, next(), &ff, &x, &lw.ff1_w2, Rows::T, 2, 0.5));
            // x += attn(ln(x))
            steps.push(ln(1, &x, next()));
            steps.push(self.gemm_fv_step(&params, next(), &h, &qb, &lw.q, Rows::T, 0, 0.0));
            steps.push(self.gemm_fv_step(&params, next(), &h, &kb, &lw.k, Rows::T, 0, 0.0));
            steps.push(self.gemm_fv_step(&params, next(), &h, &vb, &lw.v, Rows::T, 0, 0.0));
            steps.push(self.step(K::Scores, &params, next(), &[(1, &qb), (2, &kb), (3, &pos[l]), (4, &lw.bias_u), (5, &lw.bias_v), (6, &sc)], Op::Scores));
            steps.push(self.step(K::Softmax, &params, next(), &[(1, &sc)], Op::Softmax));
            steps.push(self.step(K::AttnAv, &params, next(), &[(1, &sc), (2, &vb), (3, &ctx)], Op::AttnAv));
            steps.push(self.gemm_fv_step(&params, next(), &ctx, &x, &lw.o, Rows::T, 2, 1.0));
            // x += conv(ln(x))
            steps.push(ln(2, &x, next()));
            steps.push(self.gemm_fv_step(&params, next(), &h, &ff, &lw.pw1, Rows::T, 0, 0.0));
            steps.push(self.step(K::Glu, &params, next(), &[(1, &ff), (2, &qb)], Op::Glu));
            steps.push(self.step(K::DwConv, &params, next(), &[(1, &qb), (2, &kb), (3, &lw.dw), (4, &lw.bn_scale), (5, &lw.bn_shift)], Op::DwConv));
            steps.push(self.gemm_fv_step(&params, next(), &kb, &x, &lw.pw2, Rows::T, 2, 1.0));
            // x += 0.5 * ff2(ln(x))
            steps.push(ln(3, &x, next()));
            steps.push(self.gemm_fv_step(&params, next(), &h, &ff, &lw.ff2_w1, Rows::T, 3, 0.0));
            steps.push(self.gemm_fv_step(&params, next(), &ff, &x, &lw.ff2_w2, Rows::T, 2, 0.5));
            // x = ln_out(x)
            let s = next();
            steps.push(self.step(K::LnInplace, &params, s, &[(1, &x), (3, &lw.ln[4].0), (4, &lw.ln[4].1)], Op::Ln));
        }
        steps.push(self.gemm_dense_step(&params, next(), &x, &out, &w.proj, Rows::T, 1));
        assert_eq!(steps.len(), n_steps);
        Ok(Plan { cap, mel, out, staging, params, steps, _keep: vec![s1, s2, x, h, ff, qb, kb, vb, ctx, sc, pe, pos_params].into_iter().chain(pos).collect() })
    }

    fn run_once(&self, params: &wgpu::Buffer, steps: &[Step], d: Dims) {
        let mut bytes = vec![0u8; steps.len() * SLOT as usize];
        let mut launches = Vec::new();
        for (i, s) in steps.iter().enumerate() {
            let (p, g) = s.op.launch(d);
            bytes[i * SLOT as usize..i * SLOT as usize + 32].copy_from_slice(bytemuck::bytes_of(&p));
            launches.push(g);
        }
        self.queue.write_buffer(params, 0, &bytes);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("once") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            for (s, g) in steps.iter().zip(&launches) {
                pass.set_pipeline(&self.pipes[s.kernel as usize]);
                pass.set_bind_group(0, &s.bg, &[]);
                pass.dispatch_workgroups(g[0], g[1], g[2]);
            }
        }
        self.queue.submit([enc.finish()]);
    }

    fn step(&self, kernel: K, params: &wgpu::Buffer, slot: usize, bufs: &[(u32, &wgpu::Buffer)], op: Op) -> Step {
        let layout = self.pipes[kernel as usize].get_bind_group_layout(0);
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: params, offset: slot as u64 * SLOT, size: wgpu::BufferSize::new(32) }),
        }];
        for (b, buf) in bufs {
            entries.push(wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() });
        }
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        Step { kernel, bg, op }
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_fv_step(&self, params: &wgpu::Buffer, slot: usize, x: &wgpu::Buffer, y: &wgpu::Buffer, m: &Fv, rows: Rows, mode: u32, alpha: f32) -> Step {
        let op = Op::Gemm { rows, k: m.cols, n: m.rows, mode, alpha };
        self.step(K::GemmFv, params, slot, &[(1, x), (2, y), (3, &m.codes), (4, &m.lut), (5, &self.w.zeros)], op)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_dense_step(&self, params: &wgpu::Buffer, slot: usize, x: &wgpu::Buffer, y: &wgpu::Buffer, m: &Dense, rows: Rows, mode: u32) -> Step {
        let op = Op::Gemm { rows, k: m.cols, n: m.rows, mode, alpha: 0.0 };
        self.step(K::GemmDense, params, slot, &[(1, x), (2, y), (3, &m.w), (5, &m.b)], op)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn pollster_block<F: std::future::Future>(f: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VT)
    }
    static VT: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VT)) };
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

async fn read_back(device: &wgpu::Device, staging: &wgpu::Buffer, bytes: u64) -> Result<Vec<f32>> {
    let slice = staging.slice(0..bytes);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    poll_wait(device);
    rx.await.map_err(|_| GpuError("map cancelled".into()))?.map_err(|e| GpuError(format!("map: {e}")))?;
    let out = {
        let view = slice.get_mapped_range().map_err(|e| GpuError(format!("mapped range: {e}")))?;
        bytemuck::cast_slice::<u8, f32>(&view).to_vec()
    };
    staging.unmap();
    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
fn poll_wait(device: &wgpu::Device) {
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
}

#[cfg(target_arch = "wasm32")]
fn poll_wait(_: &wgpu::Device) {}

fn upload(dev: &wgpu::Device, m: &Model) -> Weights {
    let st = wgpu::BufferUsages::STORAGE;
    let vec = |label: &str, v: &[f32]| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: f32_bytes(v), usage: st });
    let fv = |f: &FvMatrix| {
        let lut: Vec<f32> = f.lo.iter().zip(&f.hi).flat_map(|(a, b)| [*a, *b]).collect();
        Fv {
            codes: dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("codes"), contents: &f.codes, usage: st }),
            lut: vec("lohi", &lut),
            rows: f.rows as u32,
            cols: f.cols as u32,
        }
    };
    let dense = |q: &QMatrix, b: &[f32]| Dense { w: vec("dense", &q.dense()), b: vec("bias", b), rows: q.rows as u32, cols: q.cols as u32 };
    let s = &m.sub;
    let layers = m
        .layers
        .iter()
        .map(|l| LayerW {
            ff1_w1: fv(&l.ff1_w1),
            ff1_w2: fv(&l.ff1_w2),
            q: fv(&l.q),
            k: fv(&l.k),
            v: fv(&l.v),
            o: fv(&l.o),
            pos: fv(&l.pos),
            pw1: fv(&l.pw1),
            pw2: fv(&l.pw2),
            ff2_w1: fv(&l.ff2_w1),
            ff2_w2: fv(&l.ff2_w2),
            bias_u: vec("bias_u", &l.bias_u),
            bias_v: vec("bias_v", &l.bias_v),
            dw: vec("dw", &l.dw),
            bn_scale: vec("bn_scale", &l.bn_scale),
            bn_shift: vec("bn_shift", &l.bn_shift),
            ln: [&l.ln_ff1, &l.ln_att, &l.ln_conv, &l.ln_ff2, &l.ln_out].map(|n| (vec("ln_w", &n.w), vec("ln_b", &n.b))),
        })
        .collect();
    Weights {
        c0_w: vec("c0_w", &s.c0_w),
        c0_b: vec("c0_b", &s.c0_b),
        dw1_w: vec("dw1_w", &s.dw1_w),
        dw1_b: vec("dw1_b", &s.dw1_b),
        pw1: dense(&s.pw1, &s.pw1_b),
        dw2_w: vec("dw2_w", &s.dw2_w),
        dw2_b: vec("dw2_b", &s.dw2_b),
        pw2: dense(&s.pw2, &s.pw2_b),
        linear: dense(&s.linear, &s.linear_b),
        proj: dense(&m.enc_proj, &m.enc_proj_b),
        layers,
        zeros: vec("zeros", &[0f32; 4096]),
    }
}
