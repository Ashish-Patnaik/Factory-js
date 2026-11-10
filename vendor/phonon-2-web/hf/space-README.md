---
title: Phonon-2 Web
emoji: 🎙️
colorFrom: indigo
colorTo: pink
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Speech-to-text in your browser on WebGPU, WebGL or CPU
models:
- FermionResearch/Phonon-2
tags:
- speech-to-text
- webgpu
- webgl
- webassembly
- on-device
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Phonon-2 Web

Phonon-2 speech recognition running entirely in your browser. No server is involved and your audio never leaves the
page. The app picks the fastest of WebGPU, WebGL and the multi-threaded WebAssembly CPU engine for your device.

- **Live dictation:** press *Start listening* and talk. A pause ends each sentence.
- **Files:** drop any audio or video file, or press *Try a sample clip*, to get word timings and SRT or JSON export.

The first visit downloads the 163 MB [Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) weights once; the
browser caches them after that.

Browsers do not allow threads inside the embedded frame on this page. For the multi-threaded CPU engine, open the
[full page](https://{{SUBDOMAIN}}.static.hf.space). WebGPU and WebGL work either way.

**Source code, build instructions and verification:** [github.com/{{REPO}}](https://github.com/{{REPO}})

## Licences

The app is Apache-2.0. The [Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) weights (Fermion Research) and
the [Parakeet-TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) base model (NVIDIA) are CC BY 4.0. The
sample clip is from [LibriSpeech](https://www.openslr.org/12) (CC BY 4.0).
