# Phonon-2 Web

Speech recognition that runs entirely in the browser. This is a from-scratch Rust port of the
[Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) runtime, compiled to WebAssembly, with three compute
backends and automatic selection of the fastest one for each device.

**[Try the demo](https://huggingface.co/spaces/unvop/phonon-2-web)** ·
[full-page version](https://unvop-phonon-2-web.static.hf.space) (enables the multi-threaded CPU engine) ·
[model card](https://huggingface.co/FermionResearch/Phonon-2)

- **Private by design.** No server is involved and audio never leaves the page. The 163 MB weights are fetched once
  from Hugging Face and then served from the browser cache.
- **Live dictation** from the microphone, with partial results while you speak; a pause ends each sentence.
- **File transcription** of any length, cut at pauses into 25 to 35 s windows, with word timings and SRT or JSON export.
- **Same output as the reference.** Every backend produces token-identical transcripts to the official fp32
  implementation on the test set.

Phonon-2 is a 164 MB five-value quantisation of NVIDIA Parakeet-TDT 0.6B v3, averaging 5.21% word error rate across
the seven English sets of the Open ASR Leaderboard.

## Backends

| Backend | Runs on | Requires |
|---|---|---|
| **WebGPU** | WGSL compute shaders, via wgpu | Chrome or Edge 113+, Safari 26+ |
| **WebGL** | GLSL fragment shaders over float textures | WebGL2 with `EXT_color_buffer_float` (nearly every GPU since about 2014) |
| **CPU** | WebAssembly SIMD, multi-threaded, hardware FMA where available | Any modern browser |

### Automatic selection

**Auto** is the default. It:

1. reads what the browser exposes about the device: the WebGPU adapter and its limits, the WebGL renderer (recognising
   software renderers such as SwiftShader, llvmpipe and WARP), float render target support, CPU thread count,
   relaxed-SIMD support and cross-origin isolation;
2. rules out backends that cannot run or would be software-emulated;
3. times a 4-second probe on the remaining GPU backends and on the CPU, then picks the fastest. A GPU within 15% of the
   CPU wins, because it leaves the CPU free for the decoder and the page;
4. remembers the decision for that device and browser, so the probe only runs once.

The choice can be overridden in the Compute panel. Backends can also be hidden for testing with `?disable=webgpu` or
`?disable=webgl,webgpu`.

### Performance

Measured on an Apple M5 Pro with a 29.4 s LibriSpeech clip, steady state:

| | Chromium | Safari |
|---|---|---|
| WebGPU | 0.37 s (80× realtime) | 0.55 s (53×) |
| WebGL | 1.2 s (24×) | 1.5 s (19×) |
| CPU (16 / 8 threads) | 1.2 s (25×) | 1.9 s (16×) |

Older and low-power devices are slower. Auto exists so that each one gets whichever backend is fastest for it.

## Using the engine in your own page

The CPU engine blocks while it computes, so run it in a module worker:

```js
import init, { Phonon, initThreadPool } from './pkg/phonon_wasm.js';

await init();
await initThreadPool(navigator.hardwareConcurrency); // threaded build, cross-origin-isolated pages only
const bytes = new Uint8Array(await (await fetch(MODEL_URL)).arrayBuffer());
const phonon = Phonon.load(bytes, null, (stage, done, total) => {});
await phonon.enableGpu();                            // or phonon.enableGl(), or stay on the CPU
const result = JSON.parse(await phonon.transcribe(float32Audio16kMono));
// { text, words: [{ text, start, end }], tokens, windows, timings_ms, backend }
```

[`web/phonon-worker.js`](web/phonon-worker.js) and [`web/autoselect.js`](web/autoselect.js) are a complete worker with
backend selection and live dictation.

## Building

```bash
rustup toolchain install nightly --component rust-src --target wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129
./build-wasm.sh    # web/pkg (threads), web/pkg-relaxed (threads + FMA), web/pkg-st (single-threaded)
node serve.mjs     # http://localhost:8787, with the headers threads need
```

`web/` is then a static site. Threads need a cross-origin-isolated page (`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`). On hosts that cannot set those headers, `web/coi-sw.js` adds them with a
service worker; without them the CPU engine falls back to a single thread.

To publish the demo, `scripts/release.sh` builds, verifies and assembles the Space in `dist/space`, and
`scripts/upload.sh` uploads it with the `hf` CLI.

## Project layout

```
crates/phonon-core   container reader, log-mel front end, CPU encoder, LSTM + TDT decoder, windowing, live segmenter
crates/phonon-gpu    encoder on wgpu: WebGPU in the browser, Metal, Vulkan or D3D12 natively
crates/phonon-gl     encoder on WebGL2 (wasm only)
crates/phonon-wasm   wasm-bindgen API
crates/phonon-cli    native test bench: golden checks, transcription, profiling
web/                 the app, engine worker, backend selection, microphone AudioWorklet, test page
tools/               golden-reference generator and headless browser checks
scripts/             Space release and upload
```

## How it works

- **Weights stay compact.** Each five-value weight is one of `{0, ±lo, ±hi}` per output row. The CPU keeps 4-bit codes
  and expands an 8-row panel to f32 inside the GEMM, WebGPU unpacks the codes in shared-memory tiles, and WebGL decodes
  two 2-bit ternary planes in the shader. About 330 MB is resident, and every backend computes in f32 with the exact
  lo and hi values.
- **Relative-position attention** needs `relative_k_proj(pos_emb)` over `2T − 1` positions per layer. That depends only on
  the length, and a shorter clip's positions are a centred slice of a longer one's, so it is computed once and sliced.
- **The decoder** (a 2-layer LSTM and the TDT joint, greedy, durations 0 to 4) runs on the CPU in exact
  int8 × int16 arithmetic, batching the joint over upcoming frames while the prediction state is unchanged.
- **GPU passes are pre-recorded.** Each window is about 490 dispatches on WebGPU or 600 draws on WebGL, with one
  parameter upload per call; only the 640-wide encoder output is read back.

## Verification

[`tools/make_reference.py`](tools/make_reference.py) runs the official `fermion-research` full-precision path and saves
every intermediate tensor for the test clips. Against that reference:

- **Native CPU and GPU (Metal):** each stage is within about 2e-6 relative error, and every clip is token-identical.
  Run `cargo run --release -p phonon-cli --features gpu -- check <model> testdata [--gpu]`.
- **In the browser:** `/test.html`, or headless with `node tools/browser-check.mjs`, is token-identical on CPU, WebGPU
  and WebGL in Safari 27 and Chromium, with and without cross-origin isolation. Live dictation is exercised end to end.

Not yet measured: real 2015-era hardware, Firefox, and Windows and Linux GPUs.

## Licence

The code in this repository is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attributions.

The model weights are not part of this repository. They are downloaded at run time and remain under their own licences:

| Work | Author | Licence |
|---|---|---|
| [Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) (weights used at run time) | Fermion Research | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Parakeet-TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (base model of Phonon-2) | NVIDIA | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [LibriSpeech](https://www.openslr.org/12) (sample and test audio) | Panayotov et al. | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [`fermion-research`](https://pypi.org/project/fermion-research/) (container format reference) | Fermion Research | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers) (Parakeet architecture reference) | Hugging Face | Apache-2.0 |

Main Rust dependencies: [wgpu](https://github.com/gfx-rs/wgpu), [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen),
[rayon](https://github.com/rayon-rs/rayon) and [serde_json](https://github.com/serde-rs/json) (MIT or Apache-2.0),
[wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon) and [hound](https://github.com/ruuda/hound)
(Apache-2.0), [ruzstd](https://github.com/KillingSpark/zstd-rs) (MIT).

## References

1. Fermion Research. *Phonon-2.* 2026. https://huggingface.co/FermionResearch/Phonon-2
2. M. Sekoyan, N. R. Koluguri, N. Tadevosyan, et al. *Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and
   High-Performance Models for Multilingual ASR and AST.* arXiv:2509.14128, 2025.
3. H. Xu, F. Jia, S. Majumdar, H. Huang, S. Watanabe, B. Ginsburg. *Efficient Sequence Transduction by Jointly
   Predicting Tokens and Durations.* ICML 2023. arXiv:2304.06795.
4. D. Rekesh, N. R. Koluguri, S. Kriman, et al. *Fast Conformer with Linearly Scalable Attention for Efficient Speech
   Recognition.* ASRU 2023. arXiv:2305.05084.
5. N. R. Koluguri, M. Sekoyan, G. Zelenfroynd, et al. *Granary: Speech Recognition and Translation Dataset in 25
   European Languages.* Interspeech 2025. arXiv:2505.13404.
6. V. Panayotov, G. Chen, D. Povey, S. Khudanpur. *LibriSpeech: An ASR Corpus Based on Public Domain Audio Books.*
   ICASSP 2015.
