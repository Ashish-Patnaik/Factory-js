//! LSTM prediction network + TDT joint + the greedy TDT loop, as the reference torch loop runs it.

use crate::gemm::{matvec_i16, matvec_i16_multi, quantize_i16};
use crate::model::{Decoder, BLANK, DURATIONS, MAX_SYMBOLS_PER_STEP, PRED, VOCAB};

/// Upcoming frames whose joint logits are computed in one pass while the prediction state is unchanged. The batch
/// is dropped when a token is emitted (the state moves on), so the decode is the same as frame by frame.
const LOOKAHEAD: usize = 4;

/// One emitted token: vocabulary id, encoder frame, predicted duration (frames).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    pub id: u32,
    pub frame: u32,
    pub dur: u32,
}

/// Additive bonus on token logits, for hotwords: `bonus(state)` is indexed by token id (length >= VOCAB-1) and
/// `step(state, id)` advances the state after an emitted token.
pub trait Bias {
    fn bonus(&self, state: u32) -> Option<&[f32]>;
    fn step(&self, state: u32, id: u32) -> u32;
}

struct State {
    h: [Vec<f32>; 2],
    c: [Vec<f32>; 2],
}

impl State {
    fn zero() -> Self {
        State { h: [vec![0.0; PRED], vec![0.0; PRED]], c: [vec![0.0; PRED], vec![0.0; PRED]] }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Decoder {
    /// One prediction-network step from `state` with input token `last`: returns the projected output and the new state.
    fn predict(&self, last: usize, state: &State, gates: &mut [f32], tmp: &mut [f32]) -> (Vec<f32>, State) {
        let mut x = vec![0f32; PRED];
        let mut xq = vec![0i16; PRED];
        self.emb.row_f32(last, &mut x);
        let mut next = State::zero();
        for (l, layer) in self.lstm.iter().enumerate() {
            let s = quantize_i16(&x, &mut xq);
            matvec_i16(&layer.w_ih, &xq, s, Some(&layer.b), gates);
            let s = quantize_i16(&state.h[l], &mut xq);
            matvec_i16(&layer.w_hh, &xq, s, None, tmp);
            // PyTorch gate order: input, forget, cell, output
            for j in 0..PRED {
                let i = sigmoid(gates[j] + tmp[j]);
                let f = sigmoid(gates[PRED + j] + tmp[PRED + j]);
                let g = (gates[2 * PRED + j] + tmp[2 * PRED + j]).tanh();
                let o = sigmoid(gates[3 * PRED + j] + tmp[3 * PRED + j]);
                let c = f * state.c[l][j] + i * g;
                next.c[l][j] = c;
                next.h[l][j] = o * c.tanh();
            }
            x.copy_from_slice(&next.h[l]);
        }
        let mut out = vec![0f32; PRED];
        let s = quantize_i16(&x, &mut xq);
        matvec_i16(&self.proj, &xq, s, Some(&self.proj_b), &mut out);
        (out, next)
    }

    /// Greedy TDT decode over the projected encoder output `enc` `[t][PRED]`.
    pub fn greedy(&self, enc: &[f32], t: usize, bias: Option<&dyn Bias>) -> Vec<Token> {
        assert_eq!(enc.len(), t * PRED);
        let mut gates = vec![0f32; 4 * PRED];
        let mut tmp = vec![0f32; 4 * PRED];
        const NOUT: usize = VOCAB + DURATIONS.len();
        let mut batch = vec![0f32; LOOKAHEAD * NOUT];
        let (mut batch_from, mut batch_len) = (0usize, 0usize);
        let mut z = vec![0f32; PRED];
        let mut zq = vec![vec![0i16; PRED]; LOOKAHEAD];
        let mut zs = [0f32; LOOKAHEAD];
        let mut state = State::zero();
        let mut last = BLANK;
        // the prediction output for (last, state) and the state it would move to; recomputed only after an emission
        let (mut dec, mut pending) = self.predict(last, &state, &mut gates, &mut tmp);
        let mut tokens = Vec::new();
        let (mut frame, mut nsym, mut it) = (0usize, 0usize, 0usize);
        let mut bstate = 0u32;
        while frame < t && it < MAX_SYMBOLS_PER_STEP * t + 16 {
            if !(batch_len > 0 && frame >= batch_from && frame < batch_from + batch_len) {
                let n = LOOKAHEAD.min(t - frame);
                for b in 0..n {
                    let e = &enc[(frame + b) * PRED..(frame + b + 1) * PRED];
                    for j in 0..PRED {
                        z[j] = (e[j] + dec[j]).max(0.0);
                    }
                    zs[b] = quantize_i16(&z, &mut zq[b]);
                }
                let out = &mut batch[..n * NOUT];
                let jb = Some(&self.joint_b[..]);
                match n {
                    4 => matvec_i16_multi::<4>(&self.joint, &[&zq[0], &zq[1], &zq[2], &zq[3]], &[zs[0], zs[1], zs[2], zs[3]], jb, out),
                    3 => matvec_i16_multi::<3>(&self.joint, &[&zq[0], &zq[1], &zq[2]], &[zs[0], zs[1], zs[2]], jb, out),
                    2 => matvec_i16_multi::<2>(&self.joint, &[&zq[0], &zq[1]], &[zs[0], zs[1]], jb, out),
                    _ => matvec_i16(&self.joint, &zq[0], zs[0], jb, out),
                }
                batch_from = frame;
                batch_len = n;
            }
            let logits = &batch[(frame - batch_from) * NOUT..(frame - batch_from + 1) * NOUT];
            let bonus = bias.and_then(|b| b.bonus(bstate));
            let mut tok = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (k, &v) in logits[..VOCAB].iter().enumerate() {
                let v = v + bonus.and_then(|b| b.get(k)).copied().unwrap_or(0.0);
                if v > best {
                    best = v;
                    tok = k;
                }
            }
            let mut di = 0usize;
            for k in 1..DURATIONS.len() {
                if logits[VOCAB + k] > logits[VOCAB + di] {
                    di = k;
                }
            }
            let mut dur = DURATIONS[di];
            it += 1;
            if tok == BLANK && dur == 0 {
                dur = 1;
            }
            if tok != BLANK {
                tokens.push(Token { id: tok as u32, frame: frame as u32, dur: dur as u32 });
                last = tok;
                state = pending;
                if let Some(b) = bias {
                    bstate = b.step(bstate, tok as u32);
                }
                let next = self.predict(last, &state, &mut gates, &mut tmp);
                dec = next.0;
                pending = next.1;
                batch_len = 0;
            }
            if dur == 0 {
                nsym += 1;
                if nsym >= MAX_SYMBOLS_PER_STEP {
                    dur = 1;
                    nsym = 0;
                }
            } else {
                nsym = 0;
            }
            frame += dur;
        }
        tokens
    }
}
