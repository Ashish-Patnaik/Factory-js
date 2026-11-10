//! Tokens -> text, word timings and sub-word timings (the reference `_finish`, `words_from_tokens`, `_window_tokens`).

use crate::decoder::Token;
use crate::SAMPLE_RATE;

/// Seconds per encoder frame: subsampling 8 x hop 160 at 16 kHz.
pub const FRAME_S: f64 = 8.0 * 160.0 / SAMPLE_RATE as f64;

#[derive(Clone, Debug, PartialEq)]
pub struct Timed {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// Pieces dropped from transcripts (mlx-audio's rule): `<|...|>`, `<unk>`, `<pad>`.
pub fn is_special(piece: &str) -> bool {
    (piece.starts_with("<|") && piece.ends_with("|>")) || piece == "<unk>" || piece == "<pad>"
}

/// The pieces of `tokens` as `(text with "▁" -> " ", start s, duration s)`, specials dropped.
pub fn pieces(vocab: &[String], tokens: &[Token]) -> Vec<(String, f64, f64)> {
    tokens
        .iter()
        .filter_map(|t| {
            let p = vocab.get(t.id as usize)?;
            if is_special(p) {
                return None;
            }
            Some((p.replace('\u{2581}', " "), t.frame as f64 * FRAME_S, t.dur as f64 * FRAME_S))
        })
        .collect()
}

/// The transcript of one decoded window.
pub fn text(vocab: &[String], tokens: &[Token]) -> String {
    let s: String = tokens
        .iter()
        .filter_map(|t| vocab.get(t.id as usize))
        .filter(|p| !is_special(p))
        .map(|p| p.as_str())
        .collect();
    s.replace('\u{2581}', " ").trim().to_string()
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn is_punctuation(piece: &str) -> bool {
    !piece.is_empty() && !piece.chars().any(char::is_alphanumeric)
}

/// Merge pieces into words: a piece starting with a space starts a word unless it is only punctuation.
pub fn words(pieces: &[(String, f64, f64)], offset: f64, limit: Option<f64>) -> Vec<Timed> {
    let mut out = Vec::new();
    let mut cur: Vec<&(String, f64, f64)> = Vec::new();
    let flush = |cur: &mut Vec<&(String, f64, f64)>, out: &mut Vec<Timed>| {
        if cur.is_empty() {
            return;
        }
        let text: String = cur.iter().map(|t| t.0.as_str()).collect::<String>().trim().to_string();
        if !text.is_empty() {
            let last = cur[cur.len() - 1];
            let mut end = last.1 + last.2;
            if let Some(l) = limit {
                end = end.min(l);
            }
            out.push(Timed { text, start: round3(cur[0].1 + offset), end: round3(end.max(cur[0].1) + offset) });
        }
        cur.clear();
    };
    for p in pieces {
        let starts = p.0.starts_with(' ');
        let stripped = p.0.trim();
        if starts && !cur.is_empty() && (stripped.is_empty() || !is_punctuation(stripped)) {
            flush(&mut cur, &mut out);
        }
        cur.push(p);
    }
    flush(&mut cur, &mut out);
    out
}

/// Sub-word timings shifted to file time.
pub fn timed_pieces(pieces: &[(String, f64, f64)], offset: f64, limit: Option<f64>) -> Vec<Timed> {
    pieces
        .iter()
        .filter(|p| !p.0.trim().is_empty())
        .map(|p| {
            let mut end = p.1 + p.2;
            if let Some(l) = limit {
                end = p.1.max(end.min(l));
            }
            Timed { text: p.0.clone(), start: round3(p.1 + offset), end: round3(end + offset) }
        })
        .collect()
}
