// encoder/session.js - one encode job: wraps EncoderHost with frame batching, audio delivery
// and the final MP4 pull. This is the object the recorders/slideshow renderer talk to.
import fs from 'node:fs';
import path from 'node:path';
import { EncoderHost } from './host.js';
import { readWav } from '../tts.js';
import { buildCaptionData } from '../captions.js';

export class EncoderSession {
  constructor(cfg) {
    this.cfg = cfg;
    this.host = null;
    this.batch = [];
    this.batchLimit = 6;
  }

  static async create(cfg) {
    const s = new EncoderSession(cfg);
    s.host = await new EncoderHost().launch({ channel: cfg.encoder_channel ?? null });
    await s.host.initEncoder({
      width: cfg.width,
      height: cfg.height,
      fps: cfg.fps,
      quality: cfg.quality,
      bitrate: cfg.bitrate ?? null,
    });
    return s;
  }

  // ---- one-time setup -----------------------------------------------------
  async setFontFile(ttfPath) {
    if (ttfPath && fs.existsSync(ttfPath)) {
      await this.host.setFont(fs.readFileSync(ttfPath));
    }
  }

  async setCaptionsData(data) {
    // data: prebuilt { lines, style } from captions.buildCaptionData()
    if (data) await this.host.setCaptions(data);
  }

  async setAvatarFile(gifPath, corner, sizePct) {
    await this.host.setAvatar(fs.readFileSync(gifPath), corner, sizePct);
  }

  // ---- audio --------------------------------------------------------------
  // `wavPath`: voice-over file (16-bit mono, e.g. from the TTS). audioDelay = start_delay.
  async setAudio(wavPath, { audioDelay = 0, totalDuration = 0 } = {}) {
    if (!wavPath || !fs.existsSync(wavPath)) {
      // no narration: still deliver an empty buffer so the file gets a (silent) audio track
      await this.host.deliverAudio(new Float32Array(1), { sampleRate: 24000, audioDelay, totalDuration });
      return;
    }
    const { pcm, sampleRate } = readWav(wavPath);
    await this.host.deliverAudio(pcm, { sampleRate, audioDelay, totalDuration });
  }

  // ---- video frames -------------------------------------------------------
  // Push one captured JPEG frame; batches internally to amortize CDP round-trips.
  async pushFrame(jpegBuffer, t) {
    this.batch.push({ b64: jpegBuffer.toString('base64'), t });
    if (this.batch.length >= this.batchLimit) await this.flush();
  }

  async flush() {
    if (!this.batch.length) return;
    const batch = this.batch;
    this.batch = [];
    await this.host.addFrameBatch(batch);
  }

  // ---- tile-snapshot rendering (recorder.recordTiled) ----------------------
  // Upload the page's tile strip once (JPEG b64 + the scroll step used between tiles), then
  // compose video frames in batches directly in the encoder page - no per-frame round-trips.
  async prepareTiles(jpegs, step) { return this.host.rpc('prepareTiles', jpegs, step); }
  async renderTiledChunk({ frames }) { return this.host.rpc('renderTiledChunk', { frames }); }

  async begin() {
    await this.host.start();
  }

  // ---- finish -------------------------------------------------------------
  async finish(outPath, { onProgress } = {}) {
    await this.flush();
    const info = await this.host.finish();
    await this.host.pullToFile(outPath, onProgress);
    await this.host.close();
    return info;
  }

  async abort() {
    try { await this.host?.close(); } catch { /* ignore */ }
  }
}
