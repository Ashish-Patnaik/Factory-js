//! Long-audio windowing (the reference `segment.plan`): audio up to 35 s is one window; longer audio is cut into
//! 25-35 s windows, each cut in the middle of the longest quiet run of 50 ms blocks inside the band (ties to the run
//! nearest 30 s), or at 30 s when the band has no quiet block.

use crate::SAMPLE_RATE;

pub const SINGLE_SHOT_MAX_S: f64 = 35.0;
pub const BAND_MIN_S: f64 = 25.0;
pub const BAND_MAX_S: f64 = 35.0;
pub const TARGET_S: f64 = 30.0;
pub const BLOCK_S: f64 = 0.05;
pub const NOISE_FLOOR_RMS: f32 = 0.004;
pub const GATE_RATIO: f32 = 0.18;

fn s(x: f64) -> usize {
    (x * SAMPLE_RATE as f64) as usize
}

/// Window boundaries `[(start, end)]` in samples, contiguous, covering the whole clip.
pub fn plan(audio: &[f32]) -> Vec<(usize, usize)> {
    let n = audio.len();
    let single = s(SINGLE_SHOT_MAX_S);
    if n <= single {
        return vec![(0, n)];
    }
    let block = s(BLOCK_S);
    let n_blocks = n.div_ceil(block);
    let rms: Vec<f32> = (0..n_blocks)
        .map(|b| {
            let a = &audio[b * block..((b + 1) * block).min(n)];
            (a.iter().map(|v| v * v).sum::<f32>() / block as f32).sqrt()
        })
        .collect();
    let mut windows = Vec::new();
    let mut start = 0usize;
    while n - start > single {
        let lo = start + s(BAND_MIN_S);
        let hi = start + s(BAND_MAX_S);
        let b0 = lo.div_ceil(block);
        let b1 = hi / block;
        let peak = if b1 > start / block { rms[start / block..b1.min(n_blocks)].iter().cloned().fold(0.0f32, f32::max) } else { 0.0 };
        let gate = NOISE_FLOOR_RMS.max(GATE_RATIO * peak);
        let target = start + s(TARGET_S);
        let mut cut = target;
        let quiet: Vec<bool> = (b0..b1.min(n_blocks)).map(|b| rms[b] <= gate).collect();
        let mut best: Option<(usize, i64, usize)> = None;
        let mut i = 0;
        while i < quiet.len() {
            if !quiet[i] {
                i += 1;
                continue;
            }
            let mut j = i;
            while j < quiet.len() && quiet[j] {
                j += 1;
            }
            let mid = ((b0 + i) + (b0 + j)) * block / 2;
            let cand = (j - i, -((mid as i64 - target as i64).abs()), mid);
            if best.is_none_or(|b| cand > b) {
                best = Some(cand);
            }
            i = j;
        }
        if let Some(b) = best {
            cut = b.2;
        }
        let cut = lo.max(hi.min(cut).min(n));
        windows.push((start, cut));
        start = cut;
    }
    windows.push((start, n));
    windows
}

/// The engines' shared "nothing to decode" test.
pub fn is_silent(audio: &[f32]) -> bool {
    audio.is_empty() || audio.iter().fold(0f32, |m, v| m.max(v.abs())) < 1e-4
}
