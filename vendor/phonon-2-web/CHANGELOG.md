# Changelog

## 0.1.0 (2026-10-17)

First release.

- Rust port of the Phonon-2 runtime: container reader (`phonon-2.bps.tar.zst` or `model.fermion`), log-mel front end,
  FastConformer encoder, LSTM and TDT greedy decoder, pause-aligned windowing and a live-dictation segmenter.
- Three encoder backends, all token-identical to the official fp32 reference on the test set:
  - **CPU:** WebAssembly SIMD, multi-threaded (rayon on Web Workers), with a relaxed-SIMD FMA build where supported and
    a single-threaded fallback when the page is not cross-origin isolated.
  - **WebGPU:** WGSL compute kernels via wgpu.
  - **WebGL2:** GLSL fragment kernels over float textures.
- Automatic backend selection: device capabilities read from the browser, software renderers ruled out, a short timing
  probe on the candidates, and the decision cached per device.
- Web app: live dictation with partial results, file transcription of any length with word timings, SRT and JSON
  export, and the model cached in the browser after the first download.
- Static Hugging Face Space build (`scripts/release.sh`, `scripts/upload.sh`).
