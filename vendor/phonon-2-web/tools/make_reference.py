"""Golden reference for the Rust/WebAssembly port.

Loads Phonon-2 through the fermion package's full-precision CPU path (the container expanded to exact fp32 inside the
stock ParakeetForTDT graph) and, for every wav in testdata/, writes the intermediate tensors the port is checked against:

  <name>.mel.bin        [T_mel, 128]   normalised log-mel features
  <name>.sub.bin        [T, 1024]      subsampling output
  <name>.l0.bin         [T, 1024]      encoder layer 0 output
  <name>.enc.bin        [T, 1024]      encoder output
  <name>.proj.bin       [T, 640]       encoder_projector output
  reference.json        shapes, token ids, frames, durations and text per clip

Needs the `fermion-research` package with the Phonon-2 model downloaded (`phonon transcribe` once does that):

  pip install fermion-research soundfile torch
  python tools/make_reference.py

PHONON_MODEL_DIR overrides where the unpacked model is looked for.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("FERMION_DEVICE", "cpu")
os.environ.setdefault("FERMION_P2_CPU", "fp32")

import numpy as np
import soundfile as sf
import torch

from fermion._speech import engine_phonon2_cpu as e

MODEL_DIR = Path(os.environ.get(
    "PHONON_MODEL_DIR", Path.home() / ".cache/fermion/speech/FermionResearch__Phonon-2/model_phonon2_c4c_int6"))
ROOT = Path(__file__).resolve().parent.parent
TD = ROOT / "testdata"
OUT = TD / "ref"
OUT.mkdir(exist_ok=True)


def save(name, t):
    a = np.ascontiguousarray(t.detach().float().numpy() if torch.is_tensor(t) else t, dtype=np.float32)
    a.tofile(OUT / name)
    return list(a.shape)


speech = e.load(MODEL_DIR, profile="five-value", backend="phonon2-five-value", quiet=False)
m = speech.model
assert m is not None, "expected the fp32 graph path (FERMION_P2_CPU=fp32)"
speech._ctdt = None                                   # the torch TDT loop: plain fp32 joint, no C kernels
speech._cenc = None

caught = {}
m.encoder.subsampling.register_forward_hook(lambda mod, i, o: caught.__setitem__("sub", o))
m.encoder.layers[0].register_forward_hook(lambda mod, i, o: caught.__setitem__("l0", o))

manifest = {}
for wav in sorted(TD.glob("*.wav")):
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000 and audio.ndim == 1, (wav, sr, audio.shape)
    name = wav.stem
    with torch.inference_mode():
        feats, am = speech._log_mel(audio)
        enc = m.encoder(input_features=feats, attention_mask=am).last_hidden_state
        proj = m.encoder_projector(enc)
        text, n_ids, _, tokens = speech._decode_single(audio, 1.0)
    # the decode again, keeping ids/frames/durations (the _finish() inputs)
    ids_frames = {}
    orig_finish = speech._finish
    speech._finish = lambda ids, fr, du, T: ids_frames.update(ids=list(map(int, ids)), frames=list(map(int, fr)), durs=list(map(int, du)), T=int(T)) or orig_finish(ids, fr, du, T)
    with torch.inference_mode():
        speech._decode_single(audio, 1.0)
    speech._finish = orig_finish
    manifest[name] = {
        "samples": int(audio.size),
        "mel": save(f"{name}.mel.bin", feats[0]),
        "sub": save(f"{name}.sub.bin", caught["sub"][0]),
        "l0": save(f"{name}.l0.bin", caught["l0"][0]),
        "enc": save(f"{name}.enc.bin", enc[0]),
        "proj": save(f"{name}.proj.bin", proj[0]),
        "text": text,
        **ids_frames,
    }
    print(f"{name}: T={enc.shape[1]} tokens={len(ids_frames['ids'])} | {text}", file=sys.stderr)

(OUT / "reference.json").write_text(json.dumps(manifest, indent=1))
print(f"wrote {OUT}", file=sys.stderr)
