// Backend auto-selection, run in the engine worker.
//
// 1. Read what the browser exposes about this device: WebGPU adapter (vendor, architecture, fallback flag, limits),
//    WebGL2 renderer (unmasked where allowed; a context that only exists without `failIfMajorPerformanceCaveat` is a
//    software renderer), float render targets, CPU threads, cross-origin isolation.
// 2. Rule out what cannot work or would be slow for certain: software WebGPU/WebGL (SwiftShader, llvmpipe, WARP...),
//    missing float targets, too-small limits.
// 3. Time a short probe window on the remaining GPU candidate(s) and on the CPU, encoder only (the decoder runs on the
//    CPU whatever is chosen), and pick the fastest. A GPU within 15% of the CPU wins: it leaves the CPU free for the
//    decoder and the page.
//
// The page caches the decision per device fingerprint, so the probe runs once per device/browser.

const SOFTWARE = /swiftshader|llvmpipe|softpipe|lavapipe|software|basic render|microsoft basic|warp|mesa offscreen/i;
const PROBE_S = 4;
// a GPU this fast (x realtime, encoder only) cannot lose to any CPU we would run on: skip the CPU probe
const GPU_CLEAR_WIN = 30;
// below this, a slow WebGPU implementation might lose to WebGL on the same GPU: probe WebGL too
const WEBGPU_SLOW = 10;

/** `disable`: backends to treat as absent (the page's `?disable=webgpu,webgl`, for testing and troubleshooting). */
export async function deviceSpecs(disable = []) {
  const nav = self.navigator;
  const specs = {
    threads: nav.hardwareConcurrency || 1,
    isolated: !!self.crossOriginIsolated,
    ua: nav.userAgent,
    webgpu: null,
    webgl: null,
  };
  if (nav.gpu && !disable.includes('webgpu')) {
    try {
      const a = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (a) {
        const i = a.info || {};
        const fallback = !!(a.isFallbackAdapter ?? i.isFallbackAdapter);
        const name = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(' ').trim();
        specs.webgpu = {
          name: name || 'GPU',
          software: fallback || SOFTWARE.test(name),
          maxStorageMB: Math.round(a.limits.maxStorageBufferBindingSize / 2 ** 20),
        };
      }
    } catch { /* adapter request refused */ }
  }
  if (!disable.includes('webgl')) try {
    let gl = new OffscreenCanvas(1, 1).getContext('webgl2', { failIfMajorPerformanceCaveat: true });
    let caveat = false;
    if (!gl) {
      gl = new OffscreenCanvas(1, 1).getContext('webgl2');
      caveat = !!gl;
    }
    if (gl) {
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      const renderer = String(dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
      specs.webgl = {
        renderer,
        floatTargets: !!gl.getExtension('EXT_color_buffer_float'),
        maxTexture: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        software: caveat || SOFTWARE.test(renderer),
      };
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    }
  } catch { /* no OffscreenCanvas or WebGL2 */ }
  return specs;
}

/** GPU backends worth trying on this device, best first, plus the reasons others were ruled out. */
export function gpuCandidates(specs) {
  const out = [];
  const skipped = {};
  const g = specs.webgpu;
  if (!g) skipped.webgpu = 'not available in this browser';
  else if (g.software) skipped.webgpu = 'software adapter';
  else if (g.maxStorageMB < 64) skipped.webgpu = `storage buffers limited to ${g.maxStorageMB} MB`;
  else out.push('webgpu');
  const w = specs.webgl;
  if (!w) skipped.webgl = 'WebGL2 not available';
  else if (w.software) skipped.webgl = `software renderer (${w.renderer})`;
  else if (!w.floatTargets) skipped.webgl = 'no float render targets';
  else if (w.maxTexture < 4096) skipped.webgl = `textures limited to ${w.maxTexture}`;
  else out.push('webgl');
  return { candidates: out, skipped };
}

/** Deterministic noise at speech-like level: the encoder's cost does not depend on content. */
export function probeAudio(seconds = PROBE_S) {
  const n = seconds * 16000;
  const a = new Float32Array(n);
  let s = 12345;
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) >>> 0;
    a[i] = ((s >>> 8) / 16777216 - 0.5) * 0.1 * (1 + Math.sin(i / 800));
  }
  return a;
}

/**
 * Probe and choose. `phonon` is the loaded engine; `log(msg)` reports progress.
 * Returns {choice, reason, speeds: {backend: x realtime (encoder)}, skipped, specs}.
 */
export async function autoSelect(phonon, specs, log = () => {}) {
  const { candidates, skipped } = gpuCandidates(specs);
  const speeds = {};
  const failed = {};
  const audio = probeAudio();

  const measure = async (name) => {
    if (name === 'webgpu') await phonon.enableGpu();
    else if (name === 'webgl') phonon.enableGl();
    else phonon.useBackend('cpu');
    // first run builds pipelines / plans / the position cache; time the second
    await phonon.transcribe(audio.slice());
    const r = JSON.parse(await phonon.transcribe(audio.slice()));
    return PROBE_S / (r.timings_ms.encoder / 1000);
  };

  for (const name of candidates) {
    if (name === 'webgl' && speeds.webgpu >= WEBGPU_SLOW) break; // WebGPU is fine; leave WebGL alone
    try {
      log(`timing ${name}…`);
      speeds[name] = await measure(name);
    } catch (err) {
      failed[name] = String(err?.message || err);
      try { phonon.release(name); } catch { /* not initialised */ }
    }
  }
  const bestGpu = Object.entries(speeds).sort((a, b) => b[1] - a[1])[0];
  if (!bestGpu || bestGpu[1] < GPU_CLEAR_WIN) {
    log('timing cpu…');
    speeds.cpu = await measure('cpu');
  }

  let choice = 'cpu';
  if (bestGpu && (speeds.cpu === undefined || bestGpu[1] >= 0.85 * speeds.cpu)) choice = bestGpu[0];
  for (const name of ['webgpu', 'webgl']) if (name !== choice && speeds[name] !== undefined) phonon.release(name);
  if (choice === 'cpu') phonon.useBackend('cpu');
  else if (choice === 'webgpu') await phonon.enableGpu();
  else phonon.enableGl();

  const fmt = (v) => `${v.toFixed(0)}×`;
  const measured = Object.entries(speeds).map(([k, v]) => `${LABEL[k]} ${fmt(v)}`).join(', ');
  let reason;
  if (choice === 'cpu' && !bestGpu) {
    const why = Object.entries({ ...skipped, ...failed }).map(([k, v]) => `${LABEL[k]}: ${v}`).join('; ');
    reason = `no usable GPU (${why})`;
  } else {
    reason = `fastest on this device (encoder ${measured} realtime)`;
    if (speeds.cpu === undefined) reason += '; CPU not timed, the GPU is far ahead';
  }
  return { choice, reason, speeds, skipped: { ...skipped, ...failed }, specs };
}

/** "ANGLE (Apple, ANGLE Metal Renderer: Apple M5 Pro, Unspecified Version)" -> "Apple M5 Pro" (same rule as phonon-gl). */
export function tidyRenderer(raw = '') {
  let s = raw.trim();
  const m = s.match(/^ANGLE \((.*)\)$/);
  if (m) s = m[1].split(', ')[1] || m[1];
  const i = s.indexOf('Renderer: ');
  if (i >= 0) s = s.slice(i + 10);
  for (const cut of [' Direct3D', ' (0x', ', D3D', ' OpenGL', ' Vulkan']) {
    const j = s.indexOf(cut);
    if (j >= 0) s = s.slice(0, j);
  }
  return s.trim() || 'GPU';
}

export const LABEL = { cpu: 'CPU', webgpu: 'WebGPU', webgl: 'WebGL' };

/** Stable key for caching a decision: changes when the browser, GPU or build changes. */
export function fingerprint(specs, build) {
  return JSON.stringify([specs.ua, specs.threads, specs.isolated, specs.webgpu?.name, specs.webgpu?.software, specs.webgl?.renderer, build]);
}
