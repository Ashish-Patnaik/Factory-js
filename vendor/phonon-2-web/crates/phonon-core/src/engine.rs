//! Audio in, transcript out, on the CPU. The GPU front end in `phonon-gpu` reuses everything here except the encoder.

use crate::decoder::Token;
use crate::encoder::{self, PosCache};
use crate::mel::MelFrontend;
use crate::model::Model;
use crate::segment;
use crate::text::{self, Timed};
use crate::SAMPLE_RATE;

pub struct Engine {
    pub model: Model,
    pub mel: MelFrontend,
    pub pos: PosCache,
}

/// One decoded window.
pub struct WindowOut {
    pub start: usize,
    pub end: usize,
    pub tokens: Vec<Token>,
}

#[derive(Clone, Debug, Default)]
pub struct Transcript {
    pub text: String,
    pub words: Vec<Timed>,
    pub tokens: Vec<Timed>,
    /// `(start s, end s, text)` per decoded window
    pub windows: Vec<(f64, f64, String)>,
    pub audio_seconds: f64,
}

impl Transcript {
    /// Assemble from decoded windows (skipped silent windows carry no tokens).
    pub fn from_windows(vocab: &[String], samples: usize, wins: &[WindowOut]) -> Transcript {
        let mut out = Transcript { audio_seconds: samples as f64 / SAMPLE_RATE as f64, ..Default::default() };
        let mut texts = Vec::new();
        for w in wins {
            let t = text::text(vocab, &w.tokens);
            let offset = w.start as f64 / SAMPLE_RATE as f64;
            let limit = (w.end - w.start) as f64 / SAMPLE_RATE as f64;
            let p = text::pieces(vocab, &w.tokens);
            out.words.extend(text::words(&p, offset, Some(limit)));
            out.tokens.extend(text::timed_pieces(&p, offset, Some(limit)));
            out.windows.push((offset, w.end as f64 / SAMPLE_RATE as f64, t.clone()));
            if !t.is_empty() {
                texts.push(t);
            }
        }
        out.text = texts.join(" ");
        out
    }

    pub fn to_json(&self) -> serde_json::Value {
        let timed = |v: &[Timed]| -> Vec<serde_json::Value> {
            v.iter().map(|w| serde_json::json!({"text": w.text, "start": w.start, "end": w.end})).collect()
        };
        serde_json::json!({
            "text": self.text,
            "audio_seconds": self.audio_seconds,
            "words": timed(&self.words),
            "tokens": timed(&self.tokens),
            "windows": self.windows.iter().enumerate().map(|(i, w)| serde_json::json!({"id": i, "start": (w.0 * 1000.0).round() / 1000.0, "end": (w.1 * 1000.0).round() / 1000.0, "text": w.2})).collect::<Vec<_>>(),
        })
    }
}

impl Engine {
    pub fn new(model: Model) -> Engine {
        Engine { model, mel: MelFrontend::new(), pos: PosCache::default() }
    }

    /// Normalised log-mel features and their frame count.
    pub fn features(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        (self.mel.compute(audio), MelFrontend::frames(audio.len()))
    }

    /// Projected encoder output `[T][640]` for one window.
    pub fn encode(&mut self, audio: &[f32]) -> (Vec<f32>, usize) {
        let (mel, frames) = self.features(audio);
        encoder::encode(&self.model, &mut self.pos, &mel, frames)
    }

    pub fn decode(&self, enc: &[f32], t: usize) -> Vec<Token> {
        self.model.dec.greedy(enc, t, None)
    }

    /// One window, no segmentation (up to ~35 s).
    pub fn transcribe_window(&mut self, audio: &[f32]) -> Vec<Token> {
        if segment::is_silent(audio) {
            return Vec::new();
        }
        let (enc, t) = self.encode(audio);
        self.decode(&enc, t)
    }

    /// Any length: pause-aligned windows, one decode each.
    pub fn transcribe(&mut self, audio: &[f32]) -> Transcript {
        if segment::is_silent(audio) {
            return Transcript { audio_seconds: audio.len() as f64 / SAMPLE_RATE as f64, ..Default::default() };
        }
        let wins: Vec<WindowOut> = segment::plan(audio)
            .into_iter()
            .map(|(a, b)| WindowOut { start: a, end: b, tokens: self.transcribe_window(&audio[a..b]) })
            .collect();
        Transcript::from_windows(&self.model.vocab, audio.len(), &wins)
    }

    /// Plain text of one window (what live partials and finals show).
    pub fn text(&self, tokens: &[Token]) -> String {
        text::text(&self.model.vocab, tokens)
    }
}
