//! Phonon-2 speech recognition in pure Rust.
//!
//! A from-scratch port of the Phonon-2 runtime (a five-value quantised Parakeet-TDT 0.6B v3) that compiles both to
//! native code and to `wasm32-unknown-unknown`:
//!
//! * [`container`] reads the published `phonon-2.bps.tar.zst` archive (or a bare `model.fermion` + `config.json`) in
//!   one streaming pass and keeps the five-value encoder weights at 4 bits each.
//! * [`mel`] is the log-mel front end (pre-emphasis, 512-point STFT, slaney mel 128, per-feature normalisation).
//! * [`encoder`] is the FastConformer encoder on the CPU (SIMD + optional threads); the `phonon-gpu` crate runs the
//!   same graph on WebGPU / Metal / Vulkan.
//! * [`decoder`] is the LSTM prediction network, the TDT joint and the greedy TDT loop.
//! * [`segment`] and [`live`] carry the long-audio windowing and the live-microphone cadence of the reference CLI.
//! * [`engine`] ties them together: audio in, transcript with word timings out.

pub mod container;
pub mod decoder;
pub mod encoder;
pub mod engine;
pub mod gemm;
pub mod live;
pub mod mel;
pub mod model;
pub mod par;
pub mod prof;
pub mod segment;
pub(crate) mod simd;
pub mod tensor;
pub mod text;

pub use engine::{Engine, Transcript};
pub use model::Model;

pub const SAMPLE_RATE: usize = 16_000;

/// Errors raised while loading or running the model.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(format!("i/o: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! bail {
    ($($t:tt)*) => { return Err($crate::Error(format!($($t)*))) };
}
