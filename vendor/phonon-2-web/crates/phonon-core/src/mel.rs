//! Log-mel front end, written to match the reference arithmetic (HF ParakeetFeatureExtractor as the fermion CPU engine
//! spells it out): pre-emphasis 0.97, `torch.stft(n_fft=512, hop=160, win=400 Hann (symmetric), center, zero pad)`,
//! power spectrum, slaney mel 128, `log(x + 2^-24)`, per-feature normalisation over all frames (unbiased variance,
//! `std + 1e-5`).

use crate::model::N_MELS;

pub const N_FFT: usize = 512;
pub const HOP: usize = 160;
pub const WIN: usize = 400;
const N_BINS: usize = N_FFT / 2 + 1;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 1.0 / 16_777_216.0; // 2^-24

pub struct MelFrontend {
    /// Hann(400, symmetric) centred in 512 samples
    window: Vec<f32>,
    /// [N_MELS][N_BINS]
    filters: Vec<f32>,
    /// first and one-past-last non-zero bin of each filter
    spans: Vec<(usize, usize)>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    bitrev: Vec<usize>,
}

impl Default for MelFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl MelFrontend {
    pub fn new() -> Self {
        let mut window = vec![0f32; N_FFT];
        let off = (N_FFT - WIN) / 2;
        for n in 0..WIN {
            let v = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos();
            window[off + n] = v as f32;
        }
        let filters = mel_filters();
        let spans = (0..N_MELS)
            .map(|m| {
                let row = &filters[m * N_BINS..(m + 1) * N_BINS];
                let a = row.iter().position(|&v| v != 0.0).unwrap_or(0);
                let b = row.iter().rposition(|&v| v != 0.0).map_or(0, |i| i + 1);
                (a, b.max(a))
            })
            .collect();
        let cos = (0..N_FFT / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64).cos() as f32).collect();
        let sin = (0..N_FFT / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64).sin() as f32).collect();
        let bits = N_FFT.trailing_zeros();
        let bitrev = (0..N_FFT).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect();
        MelFrontend { window, filters, spans, cos, sin, bitrev }
    }

    /// Number of feature frames for `n` samples.
    pub fn frames(n: usize) -> usize {
        1 + n / HOP
    }

    /// Normalised log-mel features `[frames][128]` for 16 kHz mono audio.
    pub fn compute(&self, wave: &[f32]) -> Vec<f32> {
        let n = wave.len();
        let frames = Self::frames(n);
        // pre-emphasis, then centre padding of N_FFT/2 zeros each side
        let pad = N_FFT / 2;
        let mut x = vec![0f32; n + 2 * pad];
        if n > 0 {
            x[pad] = wave[0];
            for i in 1..n {
                x[pad + i] = wave[i] - PREEMPH * wave[i - 1];
            }
        }
        let mut out = vec![0f32; frames * N_MELS];
        let op = crate::par::SyncPtr(out.as_mut_ptr());
        let chunk = 64;
        crate::par::for_each(frames.div_ceil(chunk), |c| {
            let mut re = [0f32; N_FFT];
            let mut im = [0f32; N_FFT];
            let mut pow = [0f32; N_BINS];
            for f in c * chunk..(c * chunk + chunk).min(frames) {
                let s = &x[f * HOP..f * HOP + N_FFT];
                for i in 0..N_FFT {
                    re[self.bitrev[i]] = s[i] * self.window[i];
                    im[i] = 0.0;
                }
                self.fft(&mut re, &mut im);
                for k in 0..N_BINS {
                    pow[k] = re[k] * re[k] + im[k] * im[k];
                }
                let row = unsafe { std::slice::from_raw_parts_mut(op.get().add(f * N_MELS), N_MELS) };
                for (m, (v, &(a, b))) in row.iter_mut().zip(&self.spans).enumerate() {
                    let w = &self.filters[m * N_BINS..(m + 1) * N_BINS];
                    let acc: f32 = (a..b).map(|k| w[k] * pow[k]).sum();
                    *v = (acc + LOG_GUARD).ln();
                }
            }
        });
        // per-feature normalisation over time
        for m in 0..N_MELS {
            let mut mean = 0f64;
            for f in 0..frames {
                mean += out[f * N_MELS + m] as f64;
            }
            mean /= frames as f64;
            let mut var = 0f64;
            for f in 0..frames {
                let d = out[f * N_MELS + m] as f64 - mean;
                var += d * d;
            }
            var /= (frames.max(2) - 1) as f64;
            let denom = (var.sqrt() + 1e-5) as f32;
            let mean = mean as f32;
            for f in 0..frames {
                let v = &mut out[f * N_MELS + m];
                *v = (*v - mean) / denom;
            }
        }
        out
    }

    /// In-place radix-2 FFT of bit-reversed input.
    fn fft(&self, re: &mut [f32; N_FFT], im: &mut [f32; N_FFT]) {
        let mut len = 2;
        while len <= N_FFT {
            let half = len / 2;
            let step = N_FFT / len;
            for start in (0..N_FFT).step_by(len) {
                for j in 0..half {
                    let (wr, wi) = (self.cos[j * step], self.sin[j * step]);
                    let (a, b) = (start + j, start + j + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
    }
}

// Slaney mel scale: linear below 1 kHz, logarithmic above.
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;

fn log_step() -> f64 {
    6.4f64.ln() / 27.0
}

fn hz_to_mel(f: f64) -> f64 {
    if f >= MIN_LOG_HZ {
        MIN_LOG_MEL + (f.max(1e-30) / MIN_LOG_HZ).ln() / log_step()
    } else {
        f / F_SP
    }
}

fn mel_to_hz(m: f64) -> f64 {
    if m >= MIN_LOG_MEL {
        MIN_LOG_HZ * (log_step() * (m - MIN_LOG_MEL)).exp()
    } else {
        F_SP * m
    }
}

/// librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm="slaney"), [N_MELS][N_BINS].
pub fn mel_filters() -> Vec<f32> {
    let sr = crate::SAMPLE_RATE as f64;
    let fft_freqs: Vec<f64> = (0..N_BINS).map(|i| i as f64 * (sr / 2.0) / (N_BINS - 1) as f64).collect();
    let (m0, m1) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_f: Vec<f64> = (0..N_MELS + 2).map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (N_MELS + 1) as f64)).collect();
    let mut w = vec![0f32; N_MELS * N_BINS];
    for i in 0..N_MELS {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = -(mel_f[i] - f) / (mel_f[i + 1] - mel_f[i]);
            let upper = (mel_f[i + 2] - f) / (mel_f[i + 2] - mel_f[i + 1]);
            w[i * N_BINS + k] = (0f64.max(lower.min(upper)) * enorm) as f32;
        }
    }
    w
}
