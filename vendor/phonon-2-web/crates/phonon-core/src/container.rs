//! Reader for the published Phonon-2 artifacts, in one streaming pass.
//!
//! * `phonon-2.bps.tar.zst` (the Hugging Face download): zstd -> tar holding `config.json`, `model.fermion` and two
//!   manifests.
//! * `model.fermion` (`fermion-five-value-parakeet-v1`): `u64 LE header length | JSON header | records`, the header's
//!   `index` listing `{n: name, k: kind, shape, b: bytes}` in file order. Kinds:
//!   - `five_value` `[O, I]`: `O * ceil(I/5)` bytes of base-3 digits (5 per byte, digit k = (byte / 3^k) % 3; 0 = -1,
//!     1 = 0, 2 = +1), then one bit per non-zero weight in row-major order (LSB first; 1 = `hi` magnitude, 0 = `lo`),
//!     then `lo` f16[O], then `hi` f16[O].
//!   - `int6` / `int8`: values (int6: 4 values in 3 bytes, `v = u - 32`), then one f16 scale per output row.
//!   - `fp16`: raw little-endian halves.

use std::collections::HashMap;
use std::io::Read;

use crate::tensor::{f16_slice, FvMatrix, QMatrix};
use crate::{bail, Error, Result};

pub const FORMAT: &str = "fermion-five-value-parakeet-v1";
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

pub enum Tensor {
    Fv(FvMatrix),
    Q(QMatrix),
    F32 { shape: Vec<usize>, data: Vec<f32> },
}

/// Everything read from an artifact.
pub struct Artifact {
    pub tensors: HashMap<String, Tensor>,
    /// `config.json` (the NeMo-shaped config carrying the vocabulary).
    pub config: serde_json::Value,
}

/// Load progress: `(stage, done, total)`; `total` is 0 when unknown.
pub type Progress<'a> = &'a mut dyn FnMut(&str, u64, u64);

/// Read either a `.tar.zst` archive or a bare `model.fermion` (then `config` must be given).
pub fn read_artifact(bytes: &[u8], config: Option<&[u8]>, progress: Progress) -> Result<Artifact> {
    if bytes.len() >= 4 && bytes[..4] == ZSTD_MAGIC {
        let mut dec = ruzstd::decoding::StreamingDecoder::new(bytes).map_err(|e| Error(format!("zstd: {e}")))?;
        read_tar(&mut dec, progress)
    } else {
        let Some(config) = config else { bail!("a bare model.fermion needs its config.json") };
        let config = serde_json::from_slice(config).map_err(|e| Error(format!("config.json: {e}")))?;
        let total = bytes.len() as u64;
        let mut r = bytes;
        let tensors = read_container(&mut r, total, progress)?;
        Ok(Artifact { tensors, config })
    }
}

fn read_tar<R: Read>(r: &mut R, progress: Progress) -> Result<Artifact> {
    let mut config = None;
    let mut tensors = None;
    let mut long_name: Option<String> = None;
    let mut header = [0u8; 512];
    loop {
        if !read_full(r, &mut header)? || header.iter().all(|&b| b == 0) {
            break;
        }
        let size = parse_octal(&header[124..136])?;
        let kind = header[156];
        let mut name = long_name.take().unwrap_or_else(|| {
            let n = cstr(&header[0..100]);
            let prefix = if &header[257..262] == b"ustar" { cstr(&header[345..500]) } else { String::new() };
            if prefix.is_empty() {
                n
            } else {
                format!("{prefix}/{n}")
            }
        });
        let padded = size.div_ceil(512) * 512;
        match kind {
            b'L' => {
                let mut v = vec![0u8; padded as usize];
                r.read_exact(&mut v)?;
                long_name = Some(cstr(&v[..size as usize]));
                continue;
            }
            b'x' => {
                let mut v = vec![0u8; padded as usize];
                r.read_exact(&mut v)?;
                long_name = pax_path(&v[..size as usize]);
                continue;
            }
            _ => {}
        }
        if let Some(i) = name.rfind('/') {
            name = name[i + 1..].to_string();
        }
        let is_file = kind == b'0' || kind == 0;
        if is_file && name == "model.fermion" {
            let mut take = (&mut *r).take(size);
            tensors = Some(read_container(&mut take, size, progress)?);
            if take.limit() != 0 {
                bail!("model.fermion: {} unread bytes", take.limit());
            }
            skip(r, padded - size)?;
        } else if is_file && name == "config.json" {
            let mut v = vec![0u8; size as usize];
            r.read_exact(&mut v)?;
            config = Some(serde_json::from_slice(&v).map_err(|e| Error(format!("config.json: {e}")))?);
            skip(r, padded - size)?;
        } else {
            skip(r, padded)?;
        }
    }
    match (tensors, config) {
        (Some(tensors), Some(config)) => Ok(Artifact { tensors, config }),
        (None, _) => bail!("archive holds no model.fermion"),
        (_, None) => bail!("archive holds no config.json"),
    }
}

fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<bool> {
    let mut got = 0;
    while got < buf.len() {
        let k = r.read(&mut buf[got..])?;
        if k == 0 {
            if got == 0 {
                return Ok(false);
            }
            bail!("truncated archive");
        }
        got += k;
    }
    Ok(true)
}

fn skip<R: Read>(r: &mut R, mut n: u64) -> Result<()> {
    let mut buf = [0u8; 8192];
    while n > 0 {
        let k = (n as usize).min(buf.len());
        r.read_exact(&mut buf[..k])?;
        n -= k as u64;
    }
    Ok(())
}

fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

fn parse_octal(b: &[u8]) -> Result<u64> {
    if b[0] & 0x80 != 0 {
        // GNU base-256
        let mut v = 0u64;
        for &c in &b[1..] {
            v = (v << 8) | c as u64;
        }
        return Ok(v);
    }
    let s = cstr(b);
    let s = s.trim();
    if s.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(s, 8).map_err(|_| Error(format!("tar: bad size field {s:?}")))
}

fn pax_path(b: &[u8]) -> Option<String> {
    let s = String::from_utf8_lossy(b);
    s.lines().find_map(|l| l.split_once(' ').and_then(|(_, kv)| kv.strip_prefix("path=")).map(str::to_string))
}

struct Pending {
    name: String,
    rows: usize,
    cols: usize,
    blob: Vec<u8>,
}

/// Read a `model.fermion` stream into named tensors. Five-value records are converted in parallel batches.
pub fn read_container<R: Read>(r: &mut R, total: u64, progress: Progress) -> Result<HashMap<String, Tensor>> {
    let mut len8 = [0u8; 8];
    r.read_exact(&mut len8)?;
    let n = u64::from_le_bytes(len8) as usize;
    if n > 16 << 20 {
        bail!("model.fermion: implausible header length {n}");
    }
    let mut head = vec![0u8; n];
    r.read_exact(&mut head)?;
    let header: serde_json::Value = serde_json::from_slice(&head).map_err(|e| Error(format!("container header: {e}")))?;
    if header["format"] != FORMAT {
        bail!("model.fermion: format {} (expected {FORMAT})", header["format"]);
    }
    let index = header["index"].as_array().ok_or_else(|| Error("container header has no index".into()))?;
    let mut out = HashMap::with_capacity(index.len());
    let mut done = 8 + n as u64;
    let mut pending: Vec<Pending> = Vec::new();
    let flush = |pending: &mut Vec<Pending>, out: &mut HashMap<String, Tensor>| -> Result<()> {
        let converted = crate::par::map(pending.len(), |i| {
            let p = &pending[i];
            decode_five_value(&p.blob, p.rows, p.cols).map_err(|e| Error(format!("{}: {}", p.name, e.0)))
        });
        for (p, m) in pending.drain(..).zip(converted) {
            out.insert(p.name, Tensor::Fv(m?));
        }
        Ok(())
    };
    for e in index {
        let name = e["n"].as_str().ok_or_else(|| Error("index entry without a name".into()))?.to_string();
        let kind = e["k"].as_str().unwrap_or("");
        let bytes = e["b"].as_u64().ok_or_else(|| Error(format!("{name}: no byte count")))? as usize;
        let shape: Vec<usize> = e["shape"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
        let mut blob = vec![0u8; bytes];
        r.read_exact(&mut blob).map_err(|err| Error(format!("{name}: {err}")))?;
        done += bytes as u64;
        progress("weights", done, total);
        match kind {
            "five_value" => {
                if shape.len() != 2 {
                    bail!("{name}: five-value record of shape {shape:?}");
                }
                pending.push(Pending { name, rows: shape[0], cols: shape[1], blob });
                if pending.len() >= 4 * crate::par::threads().max(4) {
                    flush(&mut pending, &mut out)?;
                }
            }
            "int6" | "int8" => {
                let q = decode_int(&blob, &shape, if kind == "int6" { 6 } else { 8 }).map_err(|err| Error(format!("{name}: {}", err.0)))?;
                out.insert(name, Tensor::Q(q));
            }
            "fp16" => {
                let numel: usize = shape.iter().product();
                if blob.len() != 2 * numel {
                    bail!("{name}: fp16 record of {} bytes for shape {shape:?}", blob.len());
                }
                out.insert(name, Tensor::F32 { shape, data: f16_slice(&blob) });
            }
            other => bail!("{name}: unknown record kind {other:?}"),
        }
    }
    flush(&mut pending, &mut out)?;
    Ok(out)
}

/// Base-3 digits of every byte value, least significant first.
const TRITS: [[u8; 5]; 243] = {
    let mut t = [[0u8; 5]; 243];
    let mut b = 0;
    while b < 243 {
        let mut v = b;
        let mut k = 0;
        while k < 5 {
            t[b][k] = (v % 3) as u8;
            v /= 3;
            k += 1;
        }
        b += 1;
    }
    t
};

/// One five-value record -> 4-bit codes + per-row magnitudes.
pub fn decode_five_value(blob: &[u8], rows: usize, cols: usize) -> Result<FvMatrix> {
    if !cols.is_multiple_of(2) {
        bail!("odd column count {cols}");
    }
    let rb = cols.div_ceil(5);
    let trit_bytes = rows * rb;
    if blob.len() < trit_bytes + 4 * rows {
        bail!("five-value record too short ({} bytes)", blob.len());
    }
    let bits_end = blob.len() - 4 * rows;
    let bits = &blob[trit_bytes..bits_end];
    let lo = f16_slice(&blob[bits_end..bits_end + 2 * rows]);
    let hi = f16_slice(&blob[bits_end + 2 * rows..]);
    let mut codes = vec![0u8; rows * cols / 2];
    let mut bp = 0usize; // bit position over the non-zero weights
    for r in 0..rows {
        let src = &blob[r * rb..(r + 1) * rb];
        let dst = &mut codes[r * cols / 2..(r + 1) * cols / 2];
        let mut c = 0usize;
        for &byte in src {
            if byte >= 243 {
                bail!("invalid base-3 byte {byte} in row {r}");
            }
            for &t in &TRITS[byte as usize] {
                if c >= cols {
                    break;
                }
                let code = if t == 1 {
                    0
                } else {
                    let Some(&bb) = bits.get(bp >> 3) else { bail!("magnitude bits run out at row {r}") };
                    let is_hi = (bb >> (bp & 7)) & 1 == 1;
                    bp += 1;
                    match (is_hi, t == 0) {
                        (false, false) => crate::tensor::CODE_POS_LO,
                        (false, true) => crate::tensor::CODE_NEG_LO,
                        (true, false) => crate::tensor::CODE_POS_HI,
                        (true, true) => crate::tensor::CODE_NEG_HI,
                    }
                };
                dst[c / 2] |= code << ((c & 1) * 4);
                c += 1;
            }
        }
    }
    if bp.div_ceil(8) != bits.len() {
        bail!("{} magnitude bytes for {bp} non-zero weights", bits.len());
    }
    Ok(FvMatrix { rows, cols, lo, hi, codes })
}

/// An int6/int8 record -> int8 values + per-row scales.
pub fn decode_int(blob: &[u8], shape: &[usize], bits: u32) -> Result<QMatrix> {
    let rows = *shape.first().ok_or_else(|| Error("int record without a shape".into()))?;
    let total: usize = shape.iter().product();
    let cols = total / rows.max(1);
    if blob.len() < 2 * rows {
        bail!("int record too short");
    }
    let body = &blob[..blob.len() - 2 * rows];
    let scale = f16_slice(&blob[blob.len() - 2 * rows..]);
    let q: Vec<i8> = match bits {
        8 => {
            if body.len() < total {
                bail!("int8 body {} < {total}", body.len());
            }
            body[..total].iter().map(|&b| b as i8).collect()
        }
        6 => {
            if body.len() < total.div_ceil(4) * 3 {
                bail!("int6 body {} too short for {total} values", body.len());
            }
            let mut q = Vec::with_capacity(total + 3);
            for g in body.chunks_exact(3) {
                let (b0, b1, b2) = (g[0], g[1], g[2]);
                q.push((b0 & 63) as i8 - 32);
                q.push(((b0 >> 6) | ((b1 & 15) << 2)) as i8 - 32);
                q.push(((b1 >> 4) | ((b2 & 3) << 4)) as i8 - 32);
                q.push((b2 >> 2) as i8 - 32);
                if q.len() >= total {
                    break;
                }
            }
            q.truncate(total);
            q
        }
        _ => bail!("int{bits} is not supported"),
    };
    Ok(QMatrix { shape: shape.to_vec(), rows, cols, q, scale })
}
