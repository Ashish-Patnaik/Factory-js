//! The Phonon-2 encoder on WebGL2, for browsers without WebGPU.
//!
//! WebGL2 has no compute shaders, so every kernel is a full-screen fragment draw into an RGBA32F texture (needs
//! `EXT_color_buffer_float`). Tensors are flat vec4 arrays in 1024-texel-wide textures. Five-value weights are kept as
//! two 2-bit ternary planes (4 bits per weight, ~290 MB of RGBA32UI textures) and decoded in the shader against the
//! exact per-row lo/hi, with f32 accumulation, so results match the CPU path to float rounding.
//!
//! Only meaningful on `wasm32` (it drives a `WebGl2RenderingContext` from an `OffscreenCanvas` in the worker).

#[cfg(target_arch = "wasm32")]
mod shaders;
#[cfg(target_arch = "wasm32")]
mod imp;
#[cfg(target_arch = "wasm32")]
pub use imp::GlEncoder;
