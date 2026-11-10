//! The FastConformer encoder on the CPU.
//!
//! Layout is token-major everywhere (`[T][channels]`), so every linear and kernel-1 convolution is one [`gemm`].
//! Relative-position attention needs `relative_k_proj(pos_emb)` over `2T-1` positions per layer; that projection depends
//! only on positions, and the positions for a shorter clip are a centred slice of those of a longer one, so it is
//! computed once for the longest clip seen so far ([`PosCache`]) and sliced.

use crate::gemm::gemm;
use crate::model::{Layer, Model, Subsampling, CONV_K, D_FF, D_MODEL, HEAD_DIM, N_HEADS, N_MELS, PRED, SUB_CH};
use crate::par::{self, SyncPtr};
use crate::prof;
use crate::simd::{axpy, F4};

/// Encoder frames for `mel_frames` feature frames (three stride-2, pad-1, kernel-3 convolutions).
pub fn encoder_frames(mel_frames: usize) -> usize {
    mel_frames.div_ceil(2).div_ceil(2).div_ceil(2)
}

/// Sinusoidal relative position embeddings for relative positions `cap-1, cap-2, ..., -(cap-1)`: `[2cap-1][D_MODEL]`,
/// interleaved `[sin f0, cos f0, sin f1, cos f1, ...]` like the reference.
pub fn pos_embedding(cap: usize) -> Vec<f32> {
    let w = 2 * cap - 1;
    let half = D_MODEL / 2;
    let inv: Vec<f32> = (0..half).map(|m| 1.0 / 10000f32.powf((2 * m) as f32 / D_MODEL as f32)).collect();
    let mut pe = vec![0f32; w * D_MODEL];
    for k in 0..w {
        let p = (cap as f32 - 1.0) - k as f32;
        let row = &mut pe[k * D_MODEL..(k + 1) * D_MODEL];
        for m in 0..half {
            let f = inv[m] * p;
            row[2 * m] = (f as f64).sin() as f32;
            row[2 * m + 1] = (f as f64).cos() as f32;
        }
    }
    pe
}

/// Per-layer `relative_k_proj(pos_emb(cap))`, per head and transposed: `[N_HEADS][HEAD_DIM][stride]` with column `k`
/// holding relative position `cap - 1 - k`.
#[derive(Default)]
pub struct PosCache {
    pub cap: usize,
    pub stride: usize,
    pub layers: Vec<Vec<f32>>,
}

impl PosCache {
    /// Make sure clips of `t` encoder frames are covered (grows in steps of 64 frames, about 5 s).
    pub fn ensure(&mut self, model: &Model, t: usize) {
        if t <= self.cap && !self.layers.is_empty() {
            return;
        }
        let cap = t.max(1).div_ceil(64) * 64;
        let w = 2 * cap - 1;
        let stride = (w + 3).div_ceil(4) * 4 + 4; // room for the 4-wide tail reads
        let pe = pos_embedding(cap);
        let mut p = vec![0f32; w * D_MODEL];
        self.layers = model
            .layers
            .iter()
            .map(|layer| {
                gemm(&pe, w, &layer.pos, None, &mut p);
                let mut out = vec![0f32; N_HEADS * HEAD_DIM * stride];
                let op = SyncPtr(out.as_mut_ptr());
                par::for_each(N_HEADS * HEAD_DIM, |hd| {
                    let row = unsafe { std::slice::from_raw_parts_mut(op.get().add(hd * stride), stride) };
                    for k in 0..w {
                        row[k] = p[k * D_MODEL + hd];
                    }
                });
                out
            })
            .collect();
        self.cap = cap;
        self.stride = stride;
    }

    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|l| l.len() * 4).sum()
    }
}

/// Mel features `[mel_frames][128]` -> encoder input `[T][D_MODEL]`.
pub fn subsample(s: &Subsampling, mel: &[f32], mel_frames: usize) -> (Vec<f32>, usize) {
    assert_eq!(mel.len(), mel_frames * N_MELS);
    // conv0: 1 -> 256, 3x3 stride 2, pad 1, over (time, freq); out [h1][64][256], ReLU
    let (h1, w1) = (mel_frames.div_ceil(2), N_MELS / 2);
    let c0_t = transpose9(&s.c0_w);
    let mut a = vec![0f32; h1 * w1 * SUB_CH];
    let ap = SyncPtr(a.as_mut_ptr());
    par::for_each(h1, |y| {
        let mut v = [0f32; 9];
        for x in 0..w1 {
            for ky in 0..3 {
                for kx in 0..3 {
                    let (iy, ix) = ((2 * y + ky) as isize - 1, (2 * x + kx) as isize - 1);
                    v[ky * 3 + kx] = if iy >= 0 && (iy as usize) < mel_frames && ix >= 0 && (ix as usize) < N_MELS {
                        mel[iy as usize * N_MELS + ix as usize]
                    } else {
                        0.0
                    };
                }
            }
            let out = unsafe { std::slice::from_raw_parts_mut(ap.get().add((y * w1 + x) * SUB_CH), SUB_CH) };
            out.copy_from_slice(&s.c0_b);
            for k in 0..9 {
                axpy(v[k], &c0_t[k * SUB_CH..(k + 1) * SUB_CH], out);
            }
            for o in out.iter_mut() {
                *o = o.max(0.0);
            }
        }
    });
    let (b, h2, w2) = depthwise_s2(&a, h1, w1, &s.dw1_w, &s.dw1_b);
    let mut c = vec![0f32; h2 * w2 * SUB_CH];
    gemm(&b, h2 * w2, &s.pw1, Some(&s.pw1_b), &mut c);
    relu(&mut c);
    let (d, h3, w3) = depthwise_s2(&c, h2, w2, &s.dw2_w, &s.dw2_b);
    let mut e = vec![0f32; h3 * w3 * SUB_CH];
    gemm(&d, h3 * w3, &s.pw2, Some(&s.pw2_b), &mut e);
    relu(&mut e);
    // [T][freq][ch] -> [T][ch * 16 + freq]
    let t = h3;
    let mut f = vec![0f32; t * SUB_CH * w3];
    for tt in 0..t {
        for fr in 0..w3 {
            for ch in 0..SUB_CH {
                f[tt * SUB_CH * w3 + ch * w3 + fr] = e[(tt * w3 + fr) * SUB_CH + ch];
            }
        }
    }
    let mut out = vec![0f32; t * D_MODEL];
    gemm(&f, t, &s.linear, Some(&s.linear_b), &mut out);
    (out, t)
}

/// [C][9] -> [9][C]
fn transpose9(w: &[f32]) -> Vec<f32> {
    let c = w.len() / 9;
    let mut t = vec![0f32; w.len()];
    for ch in 0..c {
        for k in 0..9 {
            t[k * c + ch] = w[ch * 9 + k];
        }
    }
    t
}

/// Depthwise 3x3, stride 2, pad 1 over a channel-last `[h][w][256]` map.
fn depthwise_s2(a: &[f32], h: usize, w: usize, wt: &[f32], bias: &[f32]) -> (Vec<f32>, usize, usize) {
    let (ho, wo) = (h.div_ceil(2), w.div_ceil(2));
    let wt = transpose9(wt);
    let mut out = vec![0f32; ho * wo * SUB_CH];
    let op = SyncPtr(out.as_mut_ptr());
    par::for_each(ho, |y| {
        for x in 0..wo {
            let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((y * wo + x) * SUB_CH), SUB_CH) };
            o.copy_from_slice(bias);
            for ky in 0..3 {
                let iy = (2 * y + ky) as isize - 1;
                if iy < 0 || iy as usize >= h {
                    continue;
                }
                for kx in 0..3 {
                    let ix = (2 * x + kx) as isize - 1;
                    if ix < 0 || ix as usize >= w {
                        continue;
                    }
                    let src = &a[(iy as usize * w + ix as usize) * SUB_CH..][..SUB_CH];
                    let k = &wt[(ky * 3 + kx) * SUB_CH..][..SUB_CH];
                    for ch in 0..SUB_CH {
                        o[ch] += k[ch] * src[ch];
                    }
                }
            }
        }
    });
    (out, ho, wo)
}

fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scratch buffers for one encoder pass, reused across layers.
struct Scratch {
    h: Vec<f32>,
    ff: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    ctx: Vec<f32>,
    tmp: Vec<f32>,
}

impl Scratch {
    fn new(t: usize) -> Scratch {
        let d = || vec![0f32; t * D_MODEL];
        Scratch { h: d(), ff: vec![0f32; t * D_FF], q: d(), k: d(), v: d(), ctx: d(), tmp: d() }
    }
}

/// Run the 24 conformer layers over `x` [T][D_MODEL] in place.
pub fn layers(model: &Model, pos: &PosCache, x: &mut [f32], t: usize) {
    let mut s = Scratch::new(t);
    for (l, layer) in model.layers.iter().enumerate() {
        block(layer, &pos.layers[l], pos, x, t, &mut s);
    }
}

fn block(layer: &Layer, pt: &[f32], pos: &PosCache, x: &mut [f32], t: usize, s: &mut Scratch) {
    // x += 0.5 * ff1(ln(x))
    feed_forward(&layer.ln_ff1, &layer.ff1_w1, &layer.ff1_w2, x, t, s);
    // x += attn(ln(x))
    prof::time("layernorm", || layer.ln_att.apply(x, &mut s.h));
    attention(layer, pt, pos, t, s);
    prof::time("gemm.o", || gemm(&s.ctx, t, &layer.o, None, &mut s.tmp));
    prof::time("elementwise", || add(x, &s.tmp, 1.0));
    // x += conv(ln(x))
    prof::time("layernorm", || layer.ln_conv.apply(x, &mut s.h));
    conv_module(layer, t, s);
    prof::time("elementwise", || add(x, &s.tmp, 1.0));
    // x += 0.5 * ff2(ln(x))
    feed_forward(&layer.ln_ff2, &layer.ff2_w1, &layer.ff2_w2, x, t, s);
    // x = ln_out(x)
    prof::time("layernorm", || {
        s.tmp.copy_from_slice(x);
        layer.ln_out.apply(&s.tmp, x)
    });
}

fn add(x: &mut [f32], y: &[f32], a: f32) {
    let n = x.len();
    let chunk = 1 << 16;
    let xp = SyncPtr(x.as_mut_ptr());
    par::for_each(n.div_ceil(chunk), |c| {
        let (a0, a1) = (c * chunk, (c * chunk + chunk).min(n));
        let xs = unsafe { std::slice::from_raw_parts_mut(xp.get().add(a0), a1 - a0) };
        axpy(a, &y[a0..a1], xs);
    });
}

fn feed_forward(ln: &crate::model::LayerNorm, w1: &crate::tensor::FvMatrix, w2: &crate::tensor::FvMatrix, x: &mut [f32], t: usize, s: &mut Scratch) {
    prof::time("layernorm", || ln.apply(x, &mut s.h));
    prof::time("gemm.ff1", || gemm(&s.h, t, w1, None, &mut s.ff));
    prof::time("elementwise", || map_inplace(&mut s.ff, silu));
    prof::time("gemm.ff2", || gemm(&s.ff, t, w2, None, &mut s.tmp));
    prof::time("elementwise", || add(x, &s.tmp, 0.5));
}

fn map_inplace(x: &mut [f32], f: fn(f32) -> f32) {
    let n = x.len();
    let chunk = 1 << 15;
    let xp = SyncPtr(x.as_mut_ptr());
    par::for_each(n.div_ceil(chunk), |c| {
        let (a0, a1) = (c * chunk, (c * chunk + chunk).min(n));
        for i in a0..a1 {
            unsafe {
                let p = xp.get().add(i);
                *p = f(*p);
            }
        }
    });
}

/// Relative-position multi-head attention on `s.h`; context (before `o_proj`) into `s.ctx`.
fn attention(layer: &Layer, pt: &[f32], pos: &PosCache, t: usize, s: &mut Scratch) {
    prof::time("gemm.qkv", || {
        gemm(&s.h, t, &layer.q, None, &mut s.q);
        gemm(&s.h, t, &layer.k, None, &mut s.k);
        gemm(&s.h, t, &layer.v, None, &mut s.v)
    });
    prof::time("attention", || attention_core(layer, pt, pos, t, s));
}

fn attention_core(layer: &Layer, pt: &[f32], pos: &PosCache, t: usize, s: &mut Scratch) {
    // K transposed per head: [head][dim][tp], tp = t rounded up to 4 (zero padded)
    let tp = t.div_ceil(4) * 4;
    let mut kt = vec![0f32; N_HEADS * HEAD_DIM * tp];
    {
        let kp = SyncPtr(kt.as_mut_ptr());
        let k = &s.k;
        par::for_each(N_HEADS * HEAD_DIM, |hd| {
            let row = unsafe { std::slice::from_raw_parts_mut(kp.get().add(hd * tp), tp) };
            for j in 0..t {
                row[j] = k[j * D_MODEL + hd];
            }
        });
    }
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let off = pos.cap - t; // column of relative position t-1
    let qc = 8; // queries per work item
    let nq = t.div_ceil(qc);
    let cp = SyncPtr(s.ctx.as_mut_ptr());
    let (q, v) = (&s.q, &s.v);
    par::for_each(N_HEADS * nq, |item| {
        let (h, qb) = (item / nq, item % nq);
        let mut scores = vec![0f32; tp];
        let mut qu = [0f32; HEAD_DIM];
        let mut qv = [0f32; HEAD_DIM];
        for i in qb * qc..(qb * qc + qc).min(t) {
            for d in 0..HEAD_DIM {
                let qq = q[i * D_MODEL + h * HEAD_DIM + d];
                qu[d] = qq + layer.bias_u[h * HEAD_DIM + d];
                qv[d] = qq + layer.bias_v[h * HEAD_DIM + d];
            }
            scores.fill(0.0);
            // score[j] = qu . k[j] + qv . p[rel = i - j]; column of rel i-j is off + (t-1-i) + j
            let base = off + (t - 1 - i);
            unsafe {
                let sp = scores.as_mut_ptr();
                for d in 0..HEAD_DIM {
                    let (su, sv) = (F4::splat(qu[d]), F4::splat(qv[d]));
                    let kr = kt.as_ptr().add((h * HEAD_DIM + d) * tp);
                    let pr = pt.as_ptr().add((h * HEAD_DIM + d) * pos.stride + base);
                    let mut j = 0;
                    while j < tp {
                        let acc = su.mul_add(F4::load(kr.add(j)), F4::load(sp.add(j)));
                        sv.mul_add(F4::load(pr.add(j)), acc).store(sp.add(j));
                        j += 4;
                    }
                }
            }
            // softmax over the t real keys
            let mut mx = f32::NEG_INFINITY;
            for sc in scores[..t].iter_mut() {
                *sc *= scale;
                mx = mx.max(*sc);
            }
            let mut sum = 0f32;
            for sc in scores[..t].iter_mut() {
                *sc = (*sc - mx).exp();
                sum += *sc;
            }
            let inv = 1.0 / sum;
            let out = unsafe { std::slice::from_raw_parts_mut(cp.get().add(i * D_MODEL + h * HEAD_DIM), HEAD_DIM) };
            out.fill(0.0);
            for j in 0..t {
                axpy(scores[j] * inv, &v[j * D_MODEL + h * HEAD_DIM..j * D_MODEL + (h + 1) * HEAD_DIM], out);
            }
        }
    });
}

/// Convolution module on `s.h`; result into `s.tmp`.
fn conv_module(layer: &Layer, t: usize, s: &mut Scratch) {
    // pointwise 1024 -> 2048 into s.ff (as [t][2048]), GLU into s.q
    let pw = &mut s.ff[..t * 2 * D_MODEL];
    prof::time("gemm.pw1", || gemm(&s.h, t, &layer.pw1, None, pw));
    prof::time("conv", || conv_core(layer, t, s));
    prof::time("gemm.pw2", || gemm(&s.k, t, &layer.pw2, None, &mut s.tmp));
}

fn conv_core(layer: &Layer, t: usize, s: &mut Scratch) {
    let pw = &s.ff[..t * 2 * D_MODEL];
    let g = &mut s.q;
    {
        let gp = SyncPtr(g.as_mut_ptr());
        par::for_each(t, |tt| {
            let row = &pw[tt * 2 * D_MODEL..(tt + 1) * 2 * D_MODEL];
            let o = unsafe { std::slice::from_raw_parts_mut(gp.get().add(tt * D_MODEL), D_MODEL) };
            for c in 0..D_MODEL {
                o[c] = row[c] * sigmoid(row[D_MODEL + c]);
            }
        });
    }
    // depthwise conv over time (k = 9, pad 4), batch norm, SiLU into s.k
    let half = CONV_K / 2;
    let dp = SyncPtr(s.k.as_mut_ptr());
    let g = &s.q;
    par::for_each(t, |tt| {
        let o = unsafe { std::slice::from_raw_parts_mut(dp.get().add(tt * D_MODEL), D_MODEL) };
        o.fill(0.0);
        for kk in 0..CONV_K {
            let src = tt as isize + kk as isize - half as isize;
            if src < 0 || src as usize >= t {
                continue;
            }
            let row = &g[src as usize * D_MODEL..(src as usize + 1) * D_MODEL];
            for c in 0..D_MODEL {
                o[c] += layer.dw[c * CONV_K + kk] * row[c];
            }
        }
        for (c, v) in o.iter_mut().enumerate() {
            *v = silu(*v * layer.bn_scale[c] + layer.bn_shift[c]);
        }
    });
}

/// Full encoder: mel features -> projected encoder output `[T][PRED]` (the joint's encoder input).
pub fn encode(model: &Model, pos: &mut PosCache, mel: &[f32], mel_frames: usize) -> (Vec<f32>, usize) {
    let (mut x, t) = prof::time("subsample", || subsample(&model.sub, mel, mel_frames));
    prof::time("pos_cache", || pos.ensure(model, t));
    layers(model, pos, &mut x, t);
    let mut out = vec![0f32; t * PRED];
    gemm(&x, t, &model.enc_proj, Some(&model.enc_proj_b), &mut out);
    (out, t)
}

/// The encoder output before the projector, for tests: `(subsampled, after layer 0, final)`.
pub fn encode_debug(model: &Model, pos: &mut PosCache, mel: &[f32], mel_frames: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let (x0, t) = subsample(&model.sub, mel, mel_frames);
    pos.ensure(model, t);
    let mut s = Scratch::new(t);
    let mut x = x0.clone();
    block(&model.layers[0], &pos.layers[0], pos, &mut x, t, &mut s);
    let l0 = x.clone();
    for l in 1..model.layers.len() {
        block(&model.layers[l], &pos.layers[l], pos, &mut x, t, &mut s);
    }
    (x0, l0, x, t)
}
