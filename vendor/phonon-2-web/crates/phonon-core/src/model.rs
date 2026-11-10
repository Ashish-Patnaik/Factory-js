//! The Phonon-2 network (Parakeet-TDT 0.6B v3 shape) assembled from container tensors.

use std::collections::HashMap;

use crate::container::{read_artifact, Artifact, Progress, Tensor};
use crate::tensor::{FvMatrix, MatI16, QMatrix};
use crate::{bail, Error, Result};

pub const D_MODEL: usize = 1024;
pub const N_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const N_LAYERS: usize = 24;
pub const D_FF: usize = 4096;
pub const CONV_K: usize = 9;
pub const SUB_CH: usize = 256;
pub const N_MELS: usize = 128;
pub const PRED: usize = 640;
pub const VOCAB: usize = 8193; // 8192 pieces + blank
pub const BLANK: usize = 8192;
pub const DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];
pub const MAX_SYMBOLS_PER_STEP: usize = 10;
const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;

pub struct LayerNorm {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl LayerNorm {
    /// Row-wise layer norm of `x` [rows, D] into `y`.
    pub fn apply(&self, x: &[f32], y: &mut [f32]) {
        let d = self.w.len();
        let rows = x.len() / d;
        let yp = crate::par::SyncPtr(y.as_mut_ptr());
        let chunk = 16;
        crate::par::for_each(rows.div_ceil(chunk), |c| {
            for r in c * chunk..(c * chunk + chunk).min(rows) {
                let xr = &x[r * d..(r + 1) * d];
                let yr = unsafe { std::slice::from_raw_parts_mut(yp.get().add(r * d), d) };
                layer_norm_row(xr, &self.w, &self.b, yr);
            }
        });
    }
}

#[inline]
pub fn layer_norm_row(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32]) {
    let d = x.len() as f32;
    let mean = x.iter().sum::<f32>() / d;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..x.len() {
        y[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

pub struct Layer {
    pub ff1_w1: FvMatrix,
    pub ff1_w2: FvMatrix,
    pub q: FvMatrix,
    pub k: FvMatrix,
    pub v: FvMatrix,
    pub o: FvMatrix,
    pub pos: FvMatrix,
    pub bias_u: Vec<f32>,
    pub bias_v: Vec<f32>,
    pub pw1: FvMatrix,
    /// depthwise conv weights [D_MODEL][CONV_K]
    pub dw: Vec<f32>,
    /// batch norm folded to `y = x * bn_scale + bn_shift`
    pub bn_scale: Vec<f32>,
    pub bn_shift: Vec<f32>,
    pub pw2: FvMatrix,
    pub ff2_w1: FvMatrix,
    pub ff2_w2: FvMatrix,
    pub ln_ff1: LayerNorm,
    pub ln_att: LayerNorm,
    pub ln_conv: LayerNorm,
    pub ln_ff2: LayerNorm,
    pub ln_out: LayerNorm,
}

pub struct Subsampling {
    /// conv 1->256, 3x3 stride 2: [256][9]
    pub c0_w: Vec<f32>,
    pub c0_b: Vec<f32>,
    /// depthwise 3x3 stride 2: [256][9]
    pub dw1_w: Vec<f32>,
    pub dw1_b: Vec<f32>,
    pub pw1: QMatrix,
    pub pw1_b: Vec<f32>,
    pub dw2_w: Vec<f32>,
    pub dw2_b: Vec<f32>,
    pub pw2: QMatrix,
    pub pw2_b: Vec<f32>,
    /// 4096 -> 1024, input index = channel * 16 + freq
    pub linear: QMatrix,
    pub linear_b: Vec<f32>,
}

pub struct LstmLayer {
    pub w_ih: MatI16,
    pub w_hh: MatI16,
    /// b_ih + b_hh
    pub b: Vec<f32>,
}

pub struct Decoder {
    pub emb: QMatrix,
    pub lstm: [LstmLayer; 2],
    pub proj: MatI16,
    pub proj_b: Vec<f32>,
    pub joint: MatI16,
    pub joint_b: Vec<f32>,
}

pub struct Model {
    pub sub: Subsampling,
    pub layers: Vec<Layer>,
    pub enc_proj: QMatrix,
    pub enc_proj_b: Vec<f32>,
    pub dec: Decoder,
    pub vocab: Vec<String>,
}

struct Take(HashMap<String, Tensor>);

impl Take {
    fn fv(&mut self, n: &str) -> Result<FvMatrix> {
        match self.0.remove(n) {
            Some(Tensor::Fv(m)) => Ok(m),
            Some(_) => bail!("{n}: expected a five-value record"),
            None => bail!("container lacks {n}"),
        }
    }
    fn q(&mut self, n: &str) -> Result<QMatrix> {
        match self.0.remove(n) {
            Some(Tensor::Q(m)) => Ok(m),
            Some(_) => bail!("{n}: expected an int record"),
            None => bail!("container lacks {n}"),
        }
    }
    fn f(&mut self, n: &str, len: usize) -> Result<Vec<f32>> {
        let v = match self.0.remove(n) {
            Some(Tensor::F32 { data, .. }) => data,
            Some(Tensor::Q(m)) => m.dense(),
            Some(Tensor::Fv(_)) => bail!("{n}: unexpected five-value record"),
            None => bail!("container lacks {n}"),
        };
        if v.len() != len {
            bail!("{n}: {} values, expected {len}", v.len());
        }
        Ok(v)
    }
    fn ln(&mut self, p: &str) -> Result<LayerNorm> {
        Ok(LayerNorm { w: self.f(&format!("{p}.weight"), D_MODEL)?, b: self.f(&format!("{p}.bias"), D_MODEL)? })
    }
}

fn check(m: &FvMatrix, rows: usize, cols: usize, n: &str) -> Result<()> {
    if m.rows != rows || m.cols != cols {
        bail!("{n}: shape [{}, {}], expected [{rows}, {cols}]", m.rows, m.cols);
    }
    Ok(())
}

impl Model {
    /// Load from the bytes of `phonon-2.bps.tar.zst`, or of a bare `model.fermion` plus its `config.json`.
    pub fn load(bytes: &[u8], config: Option<&[u8]>, progress: Progress) -> Result<Model> {
        let art = read_artifact(bytes, config, progress)?;
        Self::from_artifact(art)
    }

    pub fn from_artifact(art: Artifact) -> Result<Model> {
        let vocab = vocabulary(&art.config)?;
        let mut t = Take(art.tensors);
        let e = "encoder.subsampling";
        let sub = Subsampling {
            c0_w: t.f(&format!("{e}.layers.0.weight"), SUB_CH * 9)?,
            c0_b: t.f(&format!("{e}.layers.0.bias"), SUB_CH)?,
            dw1_w: t.f(&format!("{e}.layers.2.weight"), SUB_CH * 9)?,
            dw1_b: t.f(&format!("{e}.layers.2.bias"), SUB_CH)?,
            pw1: t.q(&format!("{e}.layers.3.weight"))?,
            pw1_b: t.f(&format!("{e}.layers.3.bias"), SUB_CH)?,
            dw2_w: t.f(&format!("{e}.layers.5.weight"), SUB_CH * 9)?,
            dw2_b: t.f(&format!("{e}.layers.5.bias"), SUB_CH)?,
            pw2: t.q(&format!("{e}.layers.6.weight"))?,
            pw2_b: t.f(&format!("{e}.layers.6.bias"), SUB_CH)?,
            linear: t.q(&format!("{e}.linear.weight"))?,
            linear_b: t.f(&format!("{e}.linear.bias"), D_MODEL)?,
        };
        if sub.linear.rows != D_MODEL || sub.linear.cols != SUB_CH * (N_MELS / 8) {
            bail!("subsampling.linear has shape {:?}", sub.linear.shape);
        }
        let mut layers = Vec::with_capacity(N_LAYERS);
        for l in 0..N_LAYERS {
            let p = format!("encoder.layers.{l}");
            let mean = t.f(&format!("{p}.conv.norm.running_mean"), D_MODEL)?;
            let var = t.f(&format!("{p}.conv.norm.running_var"), D_MODEL)?;
            let bw = t.f(&format!("{p}.conv.norm.weight"), D_MODEL)?;
            let bb = t.f(&format!("{p}.conv.norm.bias"), D_MODEL)?;
            t.0.remove(&format!("{p}.conv.norm.num_batches_tracked"));
            let bn_scale: Vec<f32> = (0..D_MODEL).map(|i| bw[i] / (var[i] + BN_EPS).sqrt()).collect();
            let bn_shift: Vec<f32> = (0..D_MODEL).map(|i| bb[i] - mean[i] * bn_scale[i]).collect();
            let layer = Layer {
                ff1_w1: t.fv(&format!("{p}.feed_forward1.linear1"))?,
                ff1_w2: t.fv(&format!("{p}.feed_forward1.linear2"))?,
                q: t.fv(&format!("{p}.self_attn.q_proj"))?,
                k: t.fv(&format!("{p}.self_attn.k_proj"))?,
                v: t.fv(&format!("{p}.self_attn.v_proj"))?,
                o: t.fv(&format!("{p}.self_attn.o_proj"))?,
                pos: t.fv(&format!("{p}.self_attn.relative_k_proj"))?,
                bias_u: t.f(&format!("{p}.self_attn.bias_u"), D_MODEL)?,
                bias_v: t.f(&format!("{p}.self_attn.bias_v"), D_MODEL)?,
                pw1: t.fv(&format!("{p}.conv.pointwise_conv1"))?,
                dw: t.f(&format!("{p}.conv.depthwise_conv.weight"), D_MODEL * CONV_K)?,
                bn_scale,
                bn_shift,
                pw2: t.fv(&format!("{p}.conv.pointwise_conv2"))?,
                ff2_w1: t.fv(&format!("{p}.feed_forward2.linear1"))?,
                ff2_w2: t.fv(&format!("{p}.feed_forward2.linear2"))?,
                ln_ff1: t.ln(&format!("{p}.norm_feed_forward1"))?,
                ln_att: t.ln(&format!("{p}.norm_self_att"))?,
                ln_conv: t.ln(&format!("{p}.norm_conv"))?,
                ln_ff2: t.ln(&format!("{p}.norm_feed_forward2"))?,
                ln_out: t.ln(&format!("{p}.norm_out"))?,
            };
            check(&layer.ff1_w1, D_FF, D_MODEL, "ff1.linear1")?;
            check(&layer.ff1_w2, D_MODEL, D_FF, "ff1.linear2")?;
            check(&layer.ff2_w1, D_FF, D_MODEL, "ff2.linear1")?;
            check(&layer.ff2_w2, D_MODEL, D_FF, "ff2.linear2")?;
            for (m, n) in [(&layer.q, "q"), (&layer.k, "k"), (&layer.v, "v"), (&layer.o, "o"), (&layer.pos, "pos"), (&layer.pw2, "pw2")] {
                check(m, D_MODEL, D_MODEL, n)?;
            }
            check(&layer.pw1, 2 * D_MODEL, D_MODEL, "pw1")?;
            layers.push(layer);
        }
        let enc_proj = t.q("encoder_projector.weight")?;
        let enc_proj_b = t.f("encoder_projector.bias", PRED)?;
        let mut lstm = Vec::new();
        for l in 0..2 {
            let bi = t.f(&format!("decoder.lstm.bias_ih_l{l}"), 4 * PRED)?;
            let bh = t.f(&format!("decoder.lstm.bias_hh_l{l}"), 4 * PRED)?;
            lstm.push(LstmLayer {
                w_ih: MatI16::from_q(&t.q(&format!("decoder.lstm.weight_ih_l{l}"))?),
                w_hh: MatI16::from_q(&t.q(&format!("decoder.lstm.weight_hh_l{l}"))?),
                b: bi.iter().zip(&bh).map(|(a, b)| a + b).collect(),
            });
        }
        let [l0, l1]: [LstmLayer; 2] = lstm.try_into().map_err(|_| Error("lstm".into()))?;
        let dec = Decoder {
            emb: t.q("decoder.embedding.weight")?,
            lstm: [l0, l1],
            proj: MatI16::from_q(&t.q("decoder.decoder_projector.weight")?),
            proj_b: t.f("decoder.decoder_projector.bias", PRED)?,
            joint: MatI16::from_q(&t.q("joint.head.weight")?),
            joint_b: t.f("joint.head.bias", VOCAB + DURATIONS.len())?,
        };
        if dec.emb.rows != VOCAB || dec.joint.rows != VOCAB + DURATIONS.len() {
            bail!("decoder vocabulary size mismatch");
        }
        if !t.0.is_empty() {
            let mut left: Vec<_> = t.0.keys().cloned().collect();
            left.sort();
            bail!("container has unexpected tensors: {:?}", &left[..left.len().min(5)]);
        }
        Ok(Model { sub, layers, enc_proj, enc_proj_b, dec, vocab })
    }

    /// Resident weight bytes.
    pub fn bytes(&self) -> usize {
        let fv: usize = self
            .layers
            .iter()
            .map(|l| {
                [&l.ff1_w1, &l.ff1_w2, &l.q, &l.k, &l.v, &l.o, &l.pos, &l.pw1, &l.pw2, &l.ff2_w1, &l.ff2_w2]
                    .iter()
                    .map(|m| m.bytes())
                    .sum::<usize>()
            })
            .sum();
        let d = &self.dec;
        let q = [&self.sub.pw1, &self.sub.pw2, &self.sub.linear, &self.enc_proj, &d.emb].iter().map(|m| m.bytes()).sum::<usize>();
        let m16 = [&d.lstm[0].w_ih, &d.lstm[0].w_hh, &d.lstm[1].w_ih, &d.lstm[1].w_hh, &d.proj, &d.joint]
            .iter()
            .map(|m| m.w.len() + 4 * m.rows)
            .sum::<usize>();
        fv + q + m16
    }
}

fn vocabulary(config: &serde_json::Value) -> Result<Vec<String>> {
    let v = config
        .get("vocabulary")
        .or_else(|| config.get("joint").and_then(|j| j.get("vocabulary")))
        .or_else(|| config.get("labels"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error("config.json has no vocabulary".into()))?;
    let vocab: Vec<String> = v.iter().map(|s| s.as_str().unwrap_or("").to_string()).collect();
    if vocab.len() != BLANK {
        bail!("vocabulary has {} pieces, expected {BLANK}", vocab.len());
    }
    Ok(vocab)
}
