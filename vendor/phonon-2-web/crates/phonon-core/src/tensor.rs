//! Weight storage.
//!
//! * [`FvMatrix`]: a five-value encoder matrix. Every weight is one of `{0, +lo, -lo, +hi, -hi}` with `lo`/`hi` per
//!   output row; it is kept as a 4-bit code per weight (row-major, two codes per byte, low nibble first), so the
//!   whole encoder is ~290 MB in memory and is expanded to exact f32 one 8-row panel at a time inside the GEMM.
//! * [`QMatrix`]: an int6/int8 matrix with one f16 scale per output row (`w = q * scale`), held as int8.

use crate::gemm::NR;

/// Five-value code -> sign/magnitude: 0 = 0, 1 = +lo, 2 = -lo, 3 = +hi, 4 = -hi.
pub const CODE_ZERO: u8 = 0;
pub const CODE_POS_LO: u8 = 1;
pub const CODE_NEG_LO: u8 = 2;
pub const CODE_POS_HI: u8 = 3;
pub const CODE_NEG_HI: u8 = 4;

#[derive(Clone)]
pub struct FvMatrix {
    pub rows: usize,
    pub cols: usize,
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
    /// `rows * cols / 2` bytes; code of `(r, c)` is nibble `c & 1` of byte `r * cols / 2 + c / 2`.
    pub codes: Vec<u8>,
}

impl FvMatrix {
    #[inline]
    pub fn code(&self, r: usize, c: usize) -> u8 {
        let b = self.codes[r * self.cols / 2 + c / 2];
        if c & 1 == 0 {
            b & 15
        } else {
            b >> 4
        }
    }

    /// The five values a code can stand for, in code order, for one row.
    #[inline]
    pub fn lut(&self, r: usize) -> [f32; 16] {
        let (lo, hi) = (self.lo[r], self.hi[r]);
        let mut t = [0f32; 16];
        t[CODE_POS_LO as usize] = lo;
        t[CODE_NEG_LO as usize] = -lo;
        t[CODE_POS_HI as usize] = hi;
        t[CODE_NEG_HI as usize] = -hi;
        t
    }

    /// The dense weight (rows x cols), for tests and tools.
    pub fn dense(&self) -> Vec<f32> {
        let mut w = vec![0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let lut = self.lut(r);
            for c in 0..self.cols {
                w[r * self.cols + c] = lut[self.code(r, c) as usize];
            }
        }
        w
    }

    pub fn bytes(&self) -> usize {
        self.codes.len() + 8 * self.rows
    }
}

#[derive(Clone)]
pub struct QMatrix {
    /// The tensor's original shape (`[out, in]`, `[out, 1, k]`, `[out, in, 1, 1]` ...).
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub q: Vec<i8>,
    pub scale: Vec<f32>,
}

impl QMatrix {
    #[inline]
    pub fn row(&self, r: usize) -> &[i8] {
        &self.q[r * self.cols..(r + 1) * self.cols]
    }

    pub fn dense(&self) -> Vec<f32> {
        let mut w = vec![0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                w[r * self.cols + c] = self.q[r * self.cols + c] as f32 * self.scale[r];
            }
        }
        w
    }

    /// One row expanded to f32.
    pub fn row_f32(&self, r: usize, out: &mut [f32]) {
        let s = self.scale[r];
        for (o, &q) in out.iter_mut().zip(self.row(r)) {
            *o = q as f32 * s;
        }
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + 4 * self.rows
    }
}

/// An int6/int8 matrix for exact integer matrix-vector products against int16 activations (the decoder's per-step
/// matvecs): int8 in memory (the decoder is bandwidth-bound), widened in registers.
pub struct MatI16 {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<i8>,
    pub scale: Vec<f32>,
}

impl MatI16 {
    pub fn from_q(q: &QMatrix) -> MatI16 {
        assert_eq!(q.cols % 32, 0, "int matvec needs a multiple of 32 columns");
        MatI16 { rows: q.rows, cols: q.cols, w: q.q.clone(), scale: q.scale.clone() }
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[i8] {
        &self.w[r * self.cols..(r + 1) * self.cols]
    }
}

/// A weight matrix the GEMM can stream as k-major panels of [`NR`] output rows.
pub trait PanelSource: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// `out[k * NR + r] = W[r0 + r][k]` for `k < cols`, `r < NR`.
    fn unpack_panel(&self, r0: usize, out: &mut [f32]);
}

impl PanelSource for FvMatrix {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn unpack_panel(&self, r0: usize, out: &mut [f32]) {
        debug_assert!(out.len() >= self.cols * NR);
        let half = self.cols / 2;
        let luts: [[f32; 16]; NR] = std::array::from_fn(|r| self.lut(r0 + r));
        let rows: [&[u8]; NR] = std::array::from_fn(|r| &self.codes[(r0 + r) * half..(r0 + r + 1) * half]);
        for kb in 0..half {
            let o = &mut out[2 * kb * NR..2 * kb * NR + 2 * NR];
            for r in 0..NR {
                let b = rows[r][kb];
                o[r] = luts[r][(b & 15) as usize];
                o[NR + r] = luts[r][(b >> 4) as usize];
            }
        }
    }
}

impl PanelSource for QMatrix {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn unpack_panel(&self, r0: usize, out: &mut [f32]) {
        for r in 0..NR {
            let s = self.scale[r0 + r];
            let row = self.row(r0 + r);
            for (k, &q) in row.iter().enumerate() {
                out[k * NR + r] = q as f32 * s;
            }
        }
    }
}

/// A plain f32 matrix (rows x cols), row-major.
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
}

impl PanelSource for DenseMatrix {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn unpack_panel(&self, r0: usize, out: &mut [f32]) {
        for r in 0..NR {
            let row = &self.w[(r0 + r) * self.cols..(r0 + r + 1) * self.cols];
            for (k, &v) in row.iter().enumerate() {
                out[k * NR + r] = v;
            }
        }
    }
}

/// IEEE half -> single.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let v = man as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -v } else { v };
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

pub fn f16_slice(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}
