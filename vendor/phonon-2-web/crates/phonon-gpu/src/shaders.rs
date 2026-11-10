//! WGSL kernels for the encoder. Every kernel takes the same 32-byte parameter block at binding 0
//! (`a..f` integers, `g`/`h` floats) from a slot of one shared uniform buffer.

pub const PARAMS: &str = r#"
struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: f32, h: f32 }
@group(0) @binding(0) var<uniform> p: P;
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
"#;

/// `Y[M][N] (op)= X[M][K] * W[N][K]^T`: 64x64 tile per workgroup, 4x4 per thread (strided by 16 so shared-memory
/// reads are conflict-free), K in steps of 16. `p.a = M, p.b = K, p.c = N, p.d = mode, p.g = alpha`.
/// Modes: 0 store, 1 +bias, 2 Y += alpha*acc, 3 silu, 4 relu(+bias), 5 relu(+bias) stored as `[M/16][N][16]`
/// (the subsampling flatten: row m = t*16+f, column n -> t*N*16 + n*16 + f).
/// `LOAD_B` fills `Bs[k][n]` for the tile; the five-value variant unpacks 4-bit codes with the row's lo/hi.
const GEMM: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<WTYPE>;
@group(0) @binding(4) var<storage, read> L: array<f32>;
@group(0) @binding(5) var<storage, read> B: array<f32>;

var<workgroup> As: array<array<f32, 64>, 16>;
var<workgroup> Bs: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let M = p.a; let K = p.b; let N = p.c;
  let m0 = wid.y * 64u; let n0 = wid.x * 64u;
  let tid = lid.y * 16u + lid.x;
  var c0 = vec4f(0.0); var c1 = vec4f(0.0); var c2 = vec4f(0.0); var c3 = vec4f(0.0);
  let ty = lid.y; let tx = lid.x;
  for (var k0 = 0u; k0 < K; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u;
      let r = e / 16u; let kk = e % 16u;
      let gm = m0 + r;
      var v = 0.0;
      if (gm < M) { v = X[gm * K + k0 + kk]; }
      As[kk][r] = v;
    }
    LOAD_B
    workgroupBarrier();
    INNER
    workgroupBarrier();
  }
  let mode = p.d;
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + ty + 16u * i;
    if (m >= M) { continue; }
    var row = c0;
    if (i == 1u) { row = c1; } else if (i == 2u) { row = c2; } else if (i == 3u) { row = c3; }
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + tx + 16u * j;
      let v = row[j];
      let o = m * N + n;
      switch mode {
        case 0u: { Y[o] = v; }
        case 1u: { Y[o] = v + B[n]; }
        case 2u: { Y[o] = Y[o] + p.g * v; }
        case 3u: { Y[o] = silu(v); }
        case 4u: { Y[o] = max(v + B[n], 0.0); }
        default: { Y[(m / 16u) * N * 16u + n * 16u + (m % 16u)] = max(v + B[n], 0.0); }
      }
    }
  }
}
"#;

const LOAD_FV: &str = r#"
    if (tid < 128u) {
      let row = tid / 2u; let half = tid % 2u;
      let n = n0 + row;
      let word = W[n * (K / 8u) + k0 / 8u + half];
      let lo = L[2u * n]; let hi = L[2u * n + 1u];
      for (var q = 0u; q < 8u; q++) {
        let c = (word >> (4u * q)) & 15u;
        let mag = select(lo, hi, c >= 3u);
        let s = select(mag, -mag, c == 2u || c == 4u);
        Bs[half * 8u + q][row] = select(0.0, s, c != 0u);
      }
    }
"#;

const LOAD_DENSE: &str = r#"
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u;
      let r = e / 16u; let kk = e % 16u;
      Bs[kk][r] = W[(n0 + r) * K + k0 + kk];
    }
"#;

/// The 16-step inner product of a 64x64 tile, register-only: four rows of four columns as `vec4` accumulators.
const INNER: &str = r#"for (var kk = 0u; kk < 16u; kk++) {
      let a = vec4f(As[kk][ty], As[kk][ty + 16u], As[kk][ty + 32u], As[kk][ty + 48u]);
      let b = vec4f(Bs[kk][tx], Bs[kk][tx + 16u], Bs[kk][tx + 32u], Bs[kk][tx + 48u]);
      c0 = fma(vec4f(a.x), b, c0);
      c1 = fma(vec4f(a.y), b, c1);
      c2 = fma(vec4f(a.z), b, c2);
      c3 = fma(vec4f(a.w), b, c3);
    }"#;

pub fn gemm_fv() -> String {
    format!("{PARAMS}{}", GEMM.replace("WTYPE", "u32").replace("LOAD_B", LOAD_FV).replace("INNER", INNER))
}

pub fn gemm_dense() -> String {
    format!("{PARAMS}{}", GEMM.replace("WTYPE", "f32").replace("LOAD_B", LOAD_DENSE).replace("INNER", INNER))
}

/// Row layer norm over 1024 features, one 256-thread workgroup per row. `p.a = rows`.
/// `INPLACE` variant normalises `X` in place.
const LAYERNORM: &str = r#"
@group(0) @binding(1) var<storage, XACCESS> X: array<f32>;
YDECL
@group(0) @binding(3) var<storage, read> G: array<f32>;
@group(0) @binding(4) var<storage, read> Bt: array<f32>;
var<workgroup> red: array<f32, 256>;

fn reduce(lid: u32, v: f32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let row = wid.x + wid.y * 65535u;
  if (row >= p.a) { return; }
  let base = row * 1024u;
  var v: array<f32, 4>;
  var s = 0.0;
  for (var i = 0u; i < 4u; i++) { v[i] = X[base + lid + 256u * i]; s += v[i]; }
  let mean = reduce(lid, s) / 1024.0;
  var q = 0.0;
  for (var i = 0u; i < 4u; i++) { let d = v[i] - mean; q += d * d; }
  let inv = inverseSqrt(reduce(lid, q) / 1024.0 + 1e-5);
  for (var i = 0u; i < 4u; i++) {
    let c = lid + 256u * i;
    OUT[base + c] = (v[i] - mean) * inv * G[c] + Bt[c];
  }
}
"#;

pub fn layernorm(inplace: bool) -> String {
    let s = if inplace {
        LAYERNORM.replace("XACCESS", "read_write").replace("YDECL", "").replace("OUT", "X")
    } else {
        LAYERNORM
            .replace("XACCESS", "read")
            .replace("YDECL", "@group(0) @binding(2) var<storage, read_write> Y: array<f32>;")
            .replace("OUT", "Y")
    };
    format!("{PARAMS}{s}")
}

/// Relative-position attention scores for a 16x16 (query, key) tile of one head:
/// `S[h][i][j] = scale * ((q_i + u) . k_j + (q_i + v) . P[cap-1-i+j])`, d in chunks of 32.
/// `p.a = T, p.b = cap, p.g = scale`.
pub fn scores() -> String {
    format!(
        "{PARAMS}{}",
        r#"
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> Kb: array<f32>;
@group(0) @binding(3) var<storage, read> Pp: array<f32>;
@group(0) @binding(4) var<storage, read> BU: array<f32>;
@group(0) @binding(5) var<storage, read> BV: array<f32>;
@group(0) @binding(6) var<storage, read_write> S: array<f32>;

var<workgroup> qu: array<array<f32, 32>, 16>;
var<workgroup> qv: array<array<f32, 32>, 16>;
var<workgroup> ks: array<array<f32, 32>, 16>;
var<workgroup> ps: array<array<f32, 32>, 31>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let T = p.a; let cap = p.b;
  let h = wid.z;
  let i0 = wid.y * 16u; let j0 = wid.x * 16u;
  let ti = lid.y; let tj = lid.x;
  let tid = ti * 16u + tj;
  let npos = 2 * i32(cap) - 1;
  let pr0 = i32(cap) - 1 - i32(i0) - 15 + i32(j0);
  var acc = 0.0;
  for (var d0 = 0u; d0 < 128u; d0 += 32u) {
    for (var r = 0u; r < 2u; r++) {
      let e = tid + r * 256u;
      let row = e / 32u; let dd = e % 32u;
      let col = h * 128u + d0 + dd;
      var q = 0.0;
      if (i0 + row < T) { q = Q[(i0 + row) * 1024u + col]; }
      qu[row][dd] = q + BU[col];
      qv[row][dd] = q + BV[col];
      var k = 0.0;
      if (j0 + row < T) { k = Kb[(j0 + row) * 1024u + col]; }
      ks[row][dd] = k;
    }
    for (var r = 0u; r < 4u; r++) {
      let e = tid + r * 256u;
      if (e < 992u) {
        let row = e / 32u; let dd = e % 32u;
        let pr = pr0 + i32(row);
        var v = 0.0;
        if (pr >= 0 && pr < npos) { v = Pp[u32(pr) * 1024u + h * 128u + d0 + dd]; }
        ps[row][dd] = v;
      }
    }
    workgroupBarrier();
    let prow = 15u - ti + tj;
    for (var dd = 0u; dd < 32u; dd++) {
      acc = fma(qu[ti][dd], ks[tj][dd], acc);
      acc = fma(qv[ti][dd], ps[prow][dd], acc);
    }
    workgroupBarrier();
  }
  let i = i0 + ti; let j = j0 + tj;
  if (i < T && j < T) { S[(h * T + i) * T + j] = acc * p.g; }
}
"#
    )
}

/// Softmax over each row of `S` (`8 * T` rows of length `T`). `p.a = T`.
pub fn softmax() -> String {
    format!(
        "{PARAMS}{}",
        r#"
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let T = p.a; let lid = lid3.x;
  let base = (wid.y * T + wid.x) * T;
  var m = -3.0e38;
  for (var j = lid; j < T; j += 256u) { m = max(m, S[base + j]); }
  red[lid] = m;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) { red[lid] = max(red[lid], red[lid + s]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lid; j < T; j += 256u) { let e = exp(S[base + j] - mx); S[base + j] = e; sum += e; }
  red[lid] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var j = lid; j < T; j += 256u) { S[base + j] *= inv; }
}
"#
    )
}

/// Attention context: `C[i][h*128+n] = sum_j S[h][i][j] V[j][h*128+n]`, a 64x64 tile per workgroup. `p.a = T`.
pub fn attn_av() -> String {
    format!(
        "{PARAMS}{}",
        r#"
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

var<workgroup> As: array<array<f32, 64>, 16>;
var<workgroup> Bs: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let T = p.a;
  let h = wid.z;
  let m0 = wid.y * 64u; let n0 = wid.x * 64u;
  let tid = lid.y * 16u + lid.x;
  var c0 = vec4f(0.0); var c1 = vec4f(0.0); var c2 = vec4f(0.0); var c3 = vec4f(0.0);
  let ty = lid.y; let tx = lid.x;
  for (var k0 = 0u; k0 < T; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u;
      let r = e / 16u; let kk = e % 16u;
      var v = 0.0;
      if (m0 + r < T && k0 + kk < T) { v = S[(h * T + m0 + r) * T + k0 + kk]; }
      As[kk][r] = v;
      let kr = e / 64u; let n = e % 64u;
      var w = 0.0;
      if (k0 + kr < T) { w = V[(k0 + kr) * 1024u + h * 128u + n0 + n]; }
      Bs[kr][n] = w;
    }
    workgroupBarrier();
    INNER
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + ty + 16u * i;
    if (m >= T) { continue; }
    var row = c0;
    if (i == 1u) { row = c1; } else if (i == 2u) { row = c2; } else if (i == 3u) { row = c3; }
    for (var j = 0u; j < 4u; j++) {
      C[m * 1024u + h * 128u + n0 + tx + 16u * j] = row[j];
    }
  }
}
"#.replace("INNER", INNER)
    )
}

/// GLU over channels: `out[t][c] = a[t][c] * sigmoid(a[t][1024 + c])`. `p.a = T * 1024`.
pub fn glu() -> String {
    format!(
        "{PARAMS}{}",
        r#"
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.x + g.y * 65535u * 256u;
  if (idx >= p.a) { return; }
  let t = idx / 1024u; let c = idx % 1024u;
  O[idx] = A[t * 2048u + c] * sigm(A[t * 2048u + 1024u + c]);
}
"#
    )
}

/// Depthwise conv over time (k = 9, pad 4) + folded batch norm + SiLU. `p.a = T`.
pub fn dwconv() -> String {
    format!(
        "{PARAMS}{}",
        r#"
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<storage, read> Wd: array<f32>;
@group(0) @binding(4) var<storage, read> Sc: array<f32>;
@group(0) @binding(5) var<storage, read> Sh: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.x + g.y * 65535u * 256u;
  let T = p.a;
  if (idx >= T * 1024u) { return; }
  let t = idx / 1024u; let c = idx % 1024u;
  var acc = 0.0;
  for (var k = 0u; k < 9u; k++) {
    let s = i32(t) + i32(k) - 4;
    if (s >= 0 && s < i32(T)) { acc = fma(Wd[c * 9u + k], G[u32(s) * 1024u + c], acc); }
  }
  O[idx] = silu(acc * Sc[c] + Sh[c]);
}
"#
    )
}

/// Subsampling front, fused: conv 1->256 3x3/2 (+bias, ReLU) evaluated on the fly under the depthwise 3x3/2 conv
/// (+bias). Out `[H2][32][256]`. `p.a = mel frames, p.b = H1, p.c = H2`.
pub fn conv0_dw1() -> String {
    format!(
        "{PARAMS}{}",
        r#"
@group(0) @binding(1) var<storage, read> Mel: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<storage, read> W0: array<f32>;
@group(0) @binding(4) var<storage, read> B0: array<f32>;
@group(0) @binding(5) var<storage, read> W1: array<f32>;
@group(0) @binding(6) var<storage, read> B1: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.x + g.y * 65535u * 256u;
  let TM = i32(p.a); let H1 = i32(p.b); let H2 = p.c;
  if (idx >= H2 * 32u * 256u) { return; }
  let c = idx % 256u; let x = i32((idx / 256u) % 32u); let y = i32(idx / 8192u);
  var acc = B1[c];
  for (var ky = 0; ky < 3; ky++) {
    let pp = 2 * y + ky - 1;
    if (pp < 0 || pp >= H1) { continue; }
    for (var kx = 0; kx < 3; kx++) {
      let q = 2 * x + kx - 1;
      if (q < 0 || q >= 64) { continue; }
      var v = B0[c];
      for (var a = 0; a < 3; a++) {
        let r = 2 * pp + a - 1;
        if (r < 0 || r >= TM) { continue; }
        for (var b = 0; b < 3; b++) {
          let s = 2 * q + b - 1;
          if (s < 0 || s >= 128) { continue; }
          v = fma(W0[c * 9u + u32(a * 3 + b)], Mel[u32(r) * 128u + u32(s)], v);
        }
      }
      acc = fma(W1[c * 9u + u32(ky * 3 + kx)], max(v, 0.0), acc);
    }
  }
  O[idx] = acc;
}
"#
    )
}

/// Depthwise 3x3 stride 2 pad 1 over channel-last `[H][W][256]` (+bias). `p.a = H, p.b = W, p.c = Ho, p.d = Wo`.
pub fn dw_s2() -> String {
    format!(
        "{PARAMS}{}",
        r#"
@group(0) @binding(1) var<storage, read> I: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<storage, read> Wd: array<f32>;
@group(0) @binding(4) var<storage, read> Bd: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let idx = g.x + g.y * 65535u * 256u;
  let H = i32(p.a); let Wi = i32(p.b); let Ho = p.c; let Wo = p.d;
  if (idx >= Ho * Wo * 256u) { return; }
  let c = idx % 256u; let x = i32((idx / 256u) % Wo); let y = i32(idx / (256u * Wo));
  var acc = Bd[c];
  for (var ky = 0; ky < 3; ky++) {
    let iy = 2 * y + ky - 1;
    if (iy < 0 || iy >= H) { continue; }
    for (var kx = 0; kx < 3; kx++) {
      let ix = 2 * x + kx - 1;
      if (ix < 0 || ix >= Wi) { continue; }
      acc = fma(Wd[c * 9u + u32(ky * 3 + kx)], I[(u32(iy) * u32(Wi) + u32(ix)) * 256u + c], acc);
    }
  }
  O[idx] = acc;
}
"#
    )
}
