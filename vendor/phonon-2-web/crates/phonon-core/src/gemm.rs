//! `Y[T, O] = X[T, K] * W[O, K]^T (+ bias)` for the encoder's linears, with `W` streamed as k-major panels of 8
//! output rows that are expanded to f32 just before use (see [`PanelSource`]).
//!
//! Work is split into (block of 64 tokens) x (group of panels) items: a token block stays hot in cache while every
//! panel of its group runs over it, and a panel is expanded once per token block. Inside an item a 4x8 register tile
//! walks the whole K dimension.

use std::cell::RefCell;

use crate::par::{self, SyncPtr};
use crate::simd::F4;
use crate::tensor::PanelSource;

/// Output rows per panel.
pub const NR: usize = 8;
/// Tokens per register tile.
const MR: usize = 4;
/// Tokens per cache block.
const TB: usize = 64;

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

pub fn gemm<W: PanelSource + ?Sized>(x: &[f32], t: usize, w: &W, bias: Option<&[f32]>, y: &mut [f32]) {
    let (k, o) = (w.cols(), w.rows());
    assert_eq!(x.len(), t * k, "gemm: x is not [{t}, {k}]");
    assert_eq!(y.len(), t * o, "gemm: y is not [{t}, {o}]");
    assert_eq!(o % NR, 0, "gemm: {o} output rows is not a multiple of {NR}");
    if let Some(b) = bias {
        assert_eq!(b.len(), o);
    }
    if t == 0 {
        return;
    }
    let panels = o / NR;
    let tblocks = t.div_ceil(TB);
    let threads = par::threads();
    let group = if threads <= 1 { panels } else { (panels * tblocks / (4 * threads)).clamp(1, 16) };
    let ngroups = panels.div_ceil(group);
    let yp = SyncPtr(y.as_mut_ptr());
    let xp = SyncPtr(x.as_ptr() as *mut f32);
    par::for_each(tblocks * ngroups, |item| {
        let (tb, g) = (item / ngroups, item % ngroups);
        let (t0, t1) = (tb * TB, (tb * TB + TB).min(t));
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            if s.len() < k * NR {
                s.resize(k * NR, 0.0);
            }
            for p in g * group..((g + 1) * group).min(panels) {
                let r0 = p * NR;
                w.unpack_panel(r0, &mut s[..k * NR]);
                let b = bias.map(|b| &b[r0..r0 + NR]);
                let mut tt = t0;
                unsafe {
                    while tt + MR <= t1 {
                        kernel_4x8(k, xp.get().add(tt * k), k, s.as_ptr(), yp.get().add(tt * o + r0), o, b);
                        tt += MR;
                    }
                    while tt < t1 {
                        kernel_1x8(k, xp.get().add(tt * k), s.as_ptr(), yp.get().add(tt * o + r0), b);
                        tt += 1;
                    }
                }
            }
        });
    });
}

#[inline(always)]
fn bias_pair(b: Option<&[f32]>) -> (F4, F4) {
    match b {
        Some(b) => unsafe { (F4::load(b.as_ptr()), F4::load(b.as_ptr().add(4))) },
        None => (F4::zero(), F4::zero()),
    }
}

#[inline(always)]
unsafe fn kernel_4x8(k: usize, x: *const f32, ldx: usize, w: *const f32, y: *mut f32, ldy: usize, b: Option<&[f32]>) {
    let mut c = [[F4::zero(); 2]; MR];
    let (x0, x1, x2, x3) = (x, x.add(ldx), x.add(2 * ldx), x.add(3 * ldx));
    for kk in 0..k {
        let w0 = F4::load(w.add(kk * NR));
        let w1 = F4::load(w.add(kk * NR + 4));
        let v0 = F4::splat(*x0.add(kk));
        let v1 = F4::splat(*x1.add(kk));
        let v2 = F4::splat(*x2.add(kk));
        let v3 = F4::splat(*x3.add(kk));
        c[0][0] = v0.mul_add(w0, c[0][0]);
        c[0][1] = v0.mul_add(w1, c[0][1]);
        c[1][0] = v1.mul_add(w0, c[1][0]);
        c[1][1] = v1.mul_add(w1, c[1][1]);
        c[2][0] = v2.mul_add(w0, c[2][0]);
        c[2][1] = v2.mul_add(w1, c[2][1]);
        c[3][0] = v3.mul_add(w0, c[3][0]);
        c[3][1] = v3.mul_add(w1, c[3][1]);
    }
    let (b0, b1) = bias_pair(b);
    for (r, cr) in c.iter().enumerate() {
        cr[0].add(b0).store(y.add(r * ldy));
        cr[1].add(b1).store(y.add(r * ldy + 4));
    }
}

#[inline(always)]
unsafe fn kernel_1x8(k: usize, x: *const f32, w: *const f32, y: *mut f32, b: Option<&[f32]>) {
    let (mut c0, mut c1) = (F4::zero(), F4::zero());
    for kk in 0..k {
        let v = F4::splat(*x.add(kk));
        c0 = v.mul_add(F4::load(w.add(kk * NR)), c0);
        c1 = v.mul_add(F4::load(w.add(kk * NR + 4)), c1);
    }
    let (b0, b1) = bias_pair(b);
    c0.add(b0).store(y);
    c1.add(b1).store(y.add(4));
}

/// `y = W x (+ bias)` for one vector against an int8 matrix, rows in parallel when there are many.
pub fn matvec_q(w: &crate::tensor::QMatrix, x: &[f32], bias: Option<&[f32]>, y: &mut [f32]) {
    assert_eq!(x.len(), w.cols);
    assert_eq!(y.len(), w.rows);
    const CHUNK: usize = 512;
    let rows = w.rows;
    let run = |r0: usize, r1: usize, y: *mut f32| {
        for r in r0..r1 {
            let v = crate::simd::dot_i8(w.row(r), x) * w.scale[r] + bias.map_or(0.0, |b| b[r]);
            unsafe { *y.add(r) = v };
        }
    };
    let yp = SyncPtr(y.as_mut_ptr());
    if rows >= 2048 && par::threads() > 1 {
        par::for_each(rows.div_ceil(CHUNK), |c| run(c * CHUNK, (c * CHUNK + CHUNK).min(rows), yp.get()));
    } else {
        run(0, rows, yp.get());
    }
}

/// `x` quantised to int16 (absmax / 32767 per vector): the activations of an int16 matvec, and their scale.
pub fn quantize_i16(x: &[f32], out: &mut [i16]) -> f32 {
    let m = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if m == 0.0 {
        out.fill(0);
        return 0.0;
    }
    let s = m / 32767.0;
    let inv = 1.0 / s;
    for (o, v) in out.iter_mut().zip(x) {
        *o = (v * inv).round() as i16;
    }
    s
}

/// `y = W x (+ bias)` with int16 weights and int16-quantised `x`; exact integer sums (|w| <= 128, so a row of up to
/// 512k columns cannot overflow), one float multiply per row. Single-threaded: per decoder step a fork/join costs
/// more than the matvec.
pub fn matvec_i16(w: &crate::tensor::MatI16, xq: &[i16], xs: f32, bias: Option<&[f32]>, y: &mut [f32]) {
    assert_eq!(xq.len(), w.cols);
    assert_eq!(y.len(), w.rows);
    for (r, out) in y.iter_mut().enumerate() {
        let acc = crate::simd::dot_i8_i16(w.row(r), xq);
        *out = acc as f32 * (w.scale[r] * xs) + bias.map_or(0.0, |b| b[r]);
    }
}

/// `matvec_i16` for `N` activation vectors at once (the TDT joint over several upcoming frames): each weight row is
/// read once for all of them. `y` is `[N][rows]`.
pub fn matvec_i16_multi<const N: usize>(w: &crate::tensor::MatI16, xq: &[&[i16]; N], xs: &[f32; N], bias: Option<&[f32]>, y: &mut [f32]) {
    let rows = w.rows;
    assert_eq!(y.len(), N * rows);
    for r in 0..rows {
        let acc = crate::simd::dot_i8_i16_multi(w.row(r), xq);
        let b = bias.map_or(0.0, |b| b[r]);
        for k in 0..N {
            y[k * rows + r] = acc[k] as f32 * (w.scale[r] * xs[k]) + b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DenseMatrix;

    #[test]
    fn gemm_matches_naive() {
        for &(t, k, o) in &[(1usize, 16usize, 8usize), (5, 32, 16), (67, 48, 24), (130, 64, 64)] {
            let x: Vec<f32> = (0..t * k).map(|i| ((i * 7919) % 97) as f32 / 97.0 - 0.5).collect();
            let w: Vec<f32> = (0..o * k).map(|i| ((i * 104_729) % 89) as f32 / 89.0 - 0.5).collect();
            let b: Vec<f32> = (0..o).map(|i| i as f32 * 0.01).collect();
            let m = DenseMatrix { rows: o, cols: k, w: w.clone() };
            let mut y = vec![0f32; t * o];
            gemm(&x, t, &m, Some(&b), &mut y);
            for i in 0..t {
                for j in 0..o {
                    let r: f32 = (0..k).map(|kk| x[i * k + kk] * w[j * k + kk]).sum::<f32>() + b[j];
                    assert!((y[i * o + j] - r).abs() < 1e-4, "t={t} k={k} o={o} ({i},{j}) {} vs {r}", y[i * o + j]);
                }
            }
        }
    }
}
