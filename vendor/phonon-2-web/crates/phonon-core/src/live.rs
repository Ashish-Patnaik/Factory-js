//! Live-microphone cadence (the reference `LiveSession`), as a state machine that asks its caller to decode.
//!
//! Feed 16 kHz mono blocks; [`LiveSegmenter::feed`] returns what to decode now: a partial (the whole in-flight segment,
//! re-decoded every 0.5 s of new audio, first after 0.35 s) or a final (the segment closed by 0.7 s of trailing quiet
//! or by reaching 30 s). The energy gate is `max(0.004 RMS, 0.18 x the segment's peak block RMS)`. Decoding is left to
//! the caller so a GPU (asynchronous) or CPU engine can serve it.

use crate::SAMPLE_RATE;

pub const FIRST_PARTIAL_S: f64 = 0.35;
pub const PARTIAL_EVERY_S: f64 = 0.50;
pub const SILENCE_CLOSE_S: f64 = 0.70;
pub const MAX_SEGMENT_S: f64 = 30.0;
pub const NOISE_FLOOR_RMS: f32 = 0.004;
pub const GATE_RATIO: f32 = 0.18;

fn s(x: f64) -> usize {
    (x * SAMPLE_RATE as f64) as usize
}

#[derive(Debug)]
pub enum LiveAction {
    /// Decode this audio (the in-flight segment so far) and show it as the current hypothesis.
    Partial(Vec<f32>),
    /// Decode this closed segment and append it to the transcript.
    Final(Vec<f32>),
}

#[derive(Default)]
pub struct LiveSegmenter {
    buf: Vec<f32>,
    voiced: bool,
    last_voice: usize,
    peak_rms: f32,
    next_partial: usize,
    pending: Vec<f32>,
}

impl LiveSegmenter {
    pub fn new() -> Self {
        let mut s = LiveSegmenter::default();
        s.reset();
        s
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.voiced = false;
        self.last_voice = 0;
        self.peak_rms = 0.0;
        self.next_partial = s(FIRST_PARTIAL_S);
    }

    /// Seconds of audio in the in-flight segment.
    pub fn segment_seconds(&self) -> f64 {
        self.buf.len() as f64 / SAMPLE_RATE as f64
    }

    /// Accept samples in any block size; the gate runs on 50 ms blocks like the reference. Returns the decodes due, in
    /// order (several when a large chunk arrives at once; callers may skip all but the last partial).
    pub fn feed(&mut self, samples: &[f32]) -> Vec<LiveAction> {
        let block = s(0.05);
        self.pending.extend_from_slice(samples);
        let mut actions = Vec::new();
        let mut off = 0;
        while self.pending.len() - off >= block {
            let b: Vec<f32> = self.pending[off..off + block].to_vec();
            off += block;
            if let Some(a) = self.feed_block(&b) {
                actions.push(a);
            }
        }
        self.pending.drain(..off);
        actions
    }

    fn feed_block(&mut self, block: &[f32]) -> Option<LiveAction> {
        self.buf.extend_from_slice(block);
        let rms = (block.iter().map(|v| v * v).sum::<f32>() / block.len() as f32).sqrt();
        self.peak_rms = self.peak_rms.max(rms);
        let gate = NOISE_FLOOR_RMS.max(GATE_RATIO * self.peak_rms);
        if rms > gate {
            self.voiced = true;
            self.last_voice = self.buf.len();
        }
        let trailing = self.buf.len() - self.last_voice;
        if self.voiced && trailing >= s(SILENCE_CLOSE_S) {
            return Some(self.finalize());
        }
        if self.buf.len() >= s(MAX_SEGMENT_S) {
            if self.voiced {
                return Some(self.finalize());
            }
            // an idle microphone: keep the last second so a word starting at the boundary keeps its onset
            let keep = self.buf[self.buf.len() - SAMPLE_RATE..].to_vec();
            self.reset();
            self.buf = keep;
            return None;
        }
        if self.voiced && self.buf.len() >= self.next_partial {
            self.next_partial = self.buf.len() + s(PARTIAL_EVERY_S);
            return Some(LiveAction::Partial(self.buf.clone()));
        }
        None
    }

    fn finalize(&mut self) -> LiveAction {
        let audio = std::mem::take(&mut self.buf);
        self.reset();
        LiveAction::Final(audio)
    }

    /// Close whatever is in flight (end of session).
    pub fn finish(&mut self) -> Option<LiveAction> {
        let voiced = self.voiced && !self.buf.is_empty();
        let out = if voiced { Some(self.finalize()) } else { None };
        self.reset();
        self.pending.clear();
        out
    }
}
