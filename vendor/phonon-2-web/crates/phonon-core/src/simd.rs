//! A 4-lane f32 vector over wasm `simd128`, AArch64 NEON or plain arrays, so every kernel is written once.
//!
//! Build for the web with `-C target-feature=+simd128`; without it the scalar fallback is used (correct, slower).

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct F4(v128);

    impl F4 {
        #[inline(always)]
        pub fn zero() -> Self {
            F4(f32x4_splat(0.0))
        }
        #[inline(always)]
        pub fn splat(x: f32) -> Self {
            F4(f32x4_splat(x))
        }
        /// Unaligned load of 4 floats.
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> Self {
            F4(v128_load(p as *const v128))
        }
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) {
            v128_store(p as *mut v128, self.0)
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            F4(f32x4_add(self.0, o.0))
        }
        /// `self * b + c`: one fused (or not, per hardware) multiply-add with relaxed SIMD, else mul + add.
        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            #[cfg(target_feature = "relaxed-simd")]
            {
                F4(f32x4_relaxed_madd(self.0, b.0, c.0))
            }
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                F4(f32x4_add(f32x4_mul(self.0, b.0), c.0))
            }
        }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            f32x4_extract_lane::<0>(self.0)
                + f32x4_extract_lane::<1>(self.0)
                + f32x4_extract_lane::<2>(self.0)
                + f32x4_extract_lane::<3>(self.0)
        }
    }

    /// 16 int8 values widened to four f32 vectors.
    #[inline(always)]
    pub unsafe fn load_i8x16(p: *const i8) -> [F4; 4] {
        let v = v128_load(p as *const v128);
        let lo = i16x8_extend_low_i8x16(v);
        let hi = i16x8_extend_high_i8x16(v);
        [
            F4(f32x4_convert_i32x4(i32x4_extend_low_i16x8(lo))),
            F4(f32x4_convert_i32x4(i32x4_extend_high_i16x8(lo))),
            F4(f32x4_convert_i32x4(i32x4_extend_low_i16x8(hi))),
            F4(f32x4_convert_i32x4(i32x4_extend_high_i16x8(hi))),
        ]
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct F4(float32x4_t);

    impl F4 {
        #[inline(always)]
        pub fn zero() -> Self {
            unsafe { F4(vdupq_n_f32(0.0)) }
        }
        #[inline(always)]
        pub fn splat(x: f32) -> Self {
            unsafe { F4(vdupq_n_f32(x)) }
        }
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> Self {
            F4(vld1q_f32(p))
        }
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) {
            vst1q_f32(p, self.0)
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            unsafe { F4(vaddq_f32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            unsafe { F4(vfmaq_f32(c.0, self.0, b.0)) }
        }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            unsafe { vaddvq_f32(self.0) }
        }
    }

    #[inline(always)]
    pub unsafe fn load_i8x16(p: *const i8) -> [F4; 4] {
        let v = vld1q_s8(p);
        let lo = vmovl_s8(vget_low_s8(v));
        let hi = vmovl_s8(vget_high_s8(v));
        [
            F4(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)))),
            F4(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)))),
            F4(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)))),
            F4(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)))),
        ]
    }
}

#[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"), target_arch = "aarch64")))]
mod imp {
    #[derive(Clone, Copy)]
    pub struct F4([f32; 4]);

    impl F4 {
        #[inline(always)]
        pub fn zero() -> Self {
            F4([0.0; 4])
        }
        #[inline(always)]
        pub fn splat(x: f32) -> Self {
            F4([x; 4])
        }
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> Self {
            F4((p as *const [f32; 4]).read_unaligned())
        }
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) {
            (p as *mut [f32; 4]).write_unaligned(self.0)
        }
        #[inline(always)]
        fn zip(self, o: Self, f: impl Fn(f32, f32) -> f32) -> Self {
            F4([f(self.0[0], o.0[0]), f(self.0[1], o.0[1]), f(self.0[2], o.0[2]), f(self.0[3], o.0[3])])
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            self.zip(o, |a, b| a + b)
        }
        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            self.zip(b, |x, y| x * y).add(c)
        }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
        }
    }

    #[inline(always)]
    pub unsafe fn load_i8x16(p: *const i8) -> [F4; 4] {
        let v = (p as *const [i8; 16]).read_unaligned();
        let f = |i: usize| F4([v[i] as f32, v[i + 1] as f32, v[i + 2] as f32, v[i + 3] as f32]);
        [f(0), f(4), f(8), f(12)]
    }
}

pub use imp::{load_i8x16, F4};

/// `sum w[i] * x[i]` for int8 weights and int16 activations (length a multiple of 32), accumulated in int32.
#[inline]
pub fn dot_i8_i16(w: &[i8], x: &[i16]) -> i32 {
    debug_assert_eq!(w.len(), x.len());
    debug_assert_eq!(w.len() % 32, 0);
    dot_imp(w, x)
}

/// `out[b] = sum w[i] * xs[b][i]` for up to 4 activation vectors against one int8 row (the row is loaded and widened
/// once); lengths a multiple of 32.
#[inline]
pub fn dot_i8_i16_multi<const N: usize>(w: &[i8], xs: &[&[i16]; N]) -> [i32; N] {
    multi_imp(w, xs)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn multi_imp<const N: usize>(w: &[i8], xs: &[&[i16]; N]) -> [i32; N] {
    use core::arch::wasm32::*;
    unsafe {
        let pw = w.as_ptr() as *const v128;
        let px: [*const v128; N] = std::array::from_fn(|b| xs[b].as_ptr() as *const v128);
        let mut acc = [i32x4_splat(0); N];
        for i in 0..w.len() / 16 {
            let a = v128_load(pw.add(i));
            let (lo, hi) = (i16x8_extend_low_i8x16(a), i16x8_extend_high_i8x16(a));
            for b in 0..N {
                acc[b] = i32x4_add(acc[b], i32x4_dot_i16x8(lo, v128_load(px[b].add(2 * i))));
                acc[b] = i32x4_add(acc[b], i32x4_dot_i16x8(hi, v128_load(px[b].add(2 * i + 1))));
            }
        }
        std::array::from_fn(|b| {
            let v = acc[b];
            i32x4_extract_lane::<0>(v) + i32x4_extract_lane::<1>(v) + i32x4_extract_lane::<2>(v) + i32x4_extract_lane::<3>(v)
        })
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn multi_imp<const N: usize>(w: &[i8], xs: &[&[i16]; N]) -> [i32; N] {
    use core::arch::aarch64::*;
    unsafe {
        let pw = w.as_ptr();
        let mut acc = [vdupq_n_s32(0); N];
        let mut i = 0;
        while i < w.len() {
            let v = vld1q_s8(pw.add(i));
            let lo = vmovl_s8(vget_low_s8(v));
            let hi = vmovl_s8(vget_high_s8(v));
            for b in 0..N {
                let px = xs[b].as_ptr();
                let (x0, x1) = (vld1q_s16(px.add(i)), vld1q_s16(px.add(i + 8)));
                acc[b] = vmlal_s16(acc[b], vget_low_s16(lo), vget_low_s16(x0));
                acc[b] = vmlal_high_s16(acc[b], lo, x0);
                acc[b] = vmlal_s16(acc[b], vget_low_s16(hi), vget_low_s16(x1));
                acc[b] = vmlal_high_s16(acc[b], hi, x1);
            }
            i += 16;
        }
        std::array::from_fn(|b| vaddvq_s32(acc[b]))
    }
}

#[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"), target_arch = "aarch64")))]
#[inline(always)]
fn multi_imp<const N: usize>(w: &[i8], xs: &[&[i16]; N]) -> [i32; N] {
    std::array::from_fn(|b| dot_imp(w, xs[b]))
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn dot_imp(w: &[i8], x: &[i16]) -> i32 {
    use core::arch::wasm32::*;
    unsafe {
        let (pw, px) = (w.as_ptr() as *const v128, x.as_ptr() as *const v128);
        let mut acc = [i32x4_splat(0); 4];
        let mut i = 0;
        while i < w.len() / 16 {
            let a = v128_load(pw.add(i));
            let b = v128_load(pw.add(i + 1));
            acc[0] = i32x4_add(acc[0], i32x4_dot_i16x8(i16x8_extend_low_i8x16(a), v128_load(px.add(2 * i))));
            acc[1] = i32x4_add(acc[1], i32x4_dot_i16x8(i16x8_extend_high_i8x16(a), v128_load(px.add(2 * i + 1))));
            acc[2] = i32x4_add(acc[2], i32x4_dot_i16x8(i16x8_extend_low_i8x16(b), v128_load(px.add(2 * i + 2))));
            acc[3] = i32x4_add(acc[3], i32x4_dot_i16x8(i16x8_extend_high_i8x16(b), v128_load(px.add(2 * i + 3))));
            i += 2;
        }
        let v = i32x4_add(i32x4_add(acc[0], acc[1]), i32x4_add(acc[2], acc[3]));
        i32x4_extract_lane::<0>(v) + i32x4_extract_lane::<1>(v) + i32x4_extract_lane::<2>(v) + i32x4_extract_lane::<3>(v)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_imp(w: &[i8], x: &[i16]) -> i32 {
    use core::arch::aarch64::*;
    unsafe {
        let (pw, px) = (w.as_ptr(), x.as_ptr());
        let mut acc = [vdupq_n_s32(0); 4];
        let mut i = 0;
        while i < w.len() {
            let v = vld1q_s8(pw.add(i));
            let lo = vmovl_s8(vget_low_s8(v));
            let hi = vmovl_s8(vget_high_s8(v));
            let (x0, x1) = (vld1q_s16(px.add(i)), vld1q_s16(px.add(i + 8)));
            acc[0] = vmlal_s16(acc[0], vget_low_s16(lo), vget_low_s16(x0));
            acc[1] = vmlal_high_s16(acc[1], lo, x0);
            acc[2] = vmlal_s16(acc[2], vget_low_s16(hi), vget_low_s16(x1));
            acc[3] = vmlal_high_s16(acc[3], hi, x1);
            i += 16;
        }
        vaddvq_s32(vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3])))
    }
}

#[cfg(not(any(all(target_arch = "wasm32", target_feature = "simd128"), target_arch = "aarch64")))]
#[inline(always)]
fn dot_imp(w: &[i8], x: &[i16]) -> i32 {
    w.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum()
}

/// `sum_k q[k] * x[k]` for int8 weights and f32 activations; `x.len()` must be a multiple of 16.
#[inline]
pub fn dot_i8(q: &[i8], x: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), x.len());
    debug_assert_eq!(x.len() % 16, 0);
    let mut acc = [F4::zero(); 4];
    unsafe {
        let (pq, px) = (q.as_ptr(), x.as_ptr());
        let mut i = 0;
        while i < x.len() {
            let w = load_i8x16(pq.add(i));
            for j in 0..4 {
                acc[j] = w[j].mul_add(F4::load(px.add(i + 4 * j)), acc[j]);
            }
            i += 16;
        }
    }
    acc[0].add(acc[1]).add(acc[2].add(acc[3])).hsum()
}

/// `y += a * x`
#[inline]
pub fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let n4 = n & !3;
    let va = F4::splat(a);
    unsafe {
        let (px, py) = (x.as_ptr(), y.as_mut_ptr());
        let mut i = 0;
        while i < n4 {
            va.mul_add(F4::load(px.add(i)), F4::load(py.add(i))).store(py.add(i));
            i += 4;
        }
    }
    for i in n4..n {
        y[i] += a * x[i];
    }
}
