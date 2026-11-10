// AudioWorklet: microphone at the device rate -> 16 kHz mono Float32 blocks of 50 ms.
//
// Resampling is a windowed-sinc (Blackman, 32 taps per side) fractional resampler with its cutoff at 7.2 kHz, so
// 44.1/48 kHz input does not alias into the band the model hears. Blocks go to the engine worker over a MessagePort
// the page transfers in (`{enginePort}` on this node's port); a level reading goes to the page every block.

const OUT_RATE = 16000;
const BLOCK = 800; // 50 ms
const HALF = 32;

class MicResampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.engine = null;
    this.port.onmessage = (e) => {
      if (e.data && e.data.enginePort) this.engine = e.data.enginePort;
    };
    this.step = sampleRate / OUT_RATE; // input samples per output sample
    const cutoff = Math.min(1, OUT_RATE / sampleRate) * 0.9; // normalised to the input Nyquist
    this.hist = new Float32Array(0);
    this.pos = HALF; // position of the next output sample in `hist` coordinates
    this.out = new Float32Array(BLOCK);
    this.fill = 0;
    this.table = new Float32Array(2 * HALF * 64 + 1); // sinc*window at 1/64-sample resolution
    for (let i = 0; i < this.table.length; i++) {
      const x = i / 64 - HALF;
      const s = x === 0 ? 1 : Math.sin(Math.PI * cutoff * x) / (Math.PI * cutoff * x);
      const w = 0.42 + 0.5 * Math.cos((Math.PI * x) / HALF) + 0.08 * Math.cos((2 * Math.PI * x) / HALF);
      this.table[i] = cutoff * s * w;
    }
  }

  kernel(d) {
    const f = (d + HALF) * 64;
    const i = Math.floor(f);
    const a = f - i;
    if (i < 0 || i + 1 >= this.table.length) return 0;
    return this.table[i] * (1 - a) + this.table[i + 1] * a;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input.length) return true;
    const n = input[0].length;
    const mono = new Float32Array(n);
    for (let c = 0; c < input.length; c++) {
      const ch = input[c];
      for (let i = 0; i < n; i++) mono[i] += ch[i] / input.length;
    }
    const h = new Float32Array(this.hist.length + n);
    h.set(this.hist);
    h.set(mono, this.hist.length);
    let pos = this.pos;
    while (pos + HALF < h.length) {
      const c = Math.floor(pos);
      const frac = pos - c;
      let acc = 0;
      for (let k = -HALF + 1; k <= HALF; k++) {
        const j = c + k;
        if (j >= 0) acc += h[j] * this.kernel(k - frac);
      }
      this.out[this.fill++] = acc;
      if (this.fill === BLOCK) {
        const block = this.out.slice();
        let s = 0;
        for (let i = 0; i < BLOCK; i++) s += block[i] * block[i];
        this.port.postMessage({ rms: Math.sqrt(s / BLOCK) });
        if (this.engine) this.engine.postMessage(block, [block.buffer]);
        this.fill = 0;
      }
      pos += this.step;
    }
    // keep enough history for the next call's taps
    const keep = Math.max(0, Math.floor(pos) - HALF);
    this.hist = h.slice(keep);
    this.pos = pos - keep;
    return true;
  }
}

registerProcessor('mic-resampler', MicResampler);
