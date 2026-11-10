//! GLSL ES 3.00 fragment kernels. Every tensor is a flat array of vec4 stored in a 1024-texel-wide RGBA32F texture
//! (texel `i` at `(i % 1024, i / 1024)`); every draw covers the output's texels, one fragment per output vec4.
//! Parameters: `ua`, `ub` (ints; `ub.w` = output texel count) and `uf` (floats).

pub const VERTEX: &str = r#"#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}
"#;

const HEAD: &str = r#"#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
uniform ivec4 ua;
uniform ivec4 ub;
uniform vec4 uf;
out vec4 o;
ivec2 at(int i) { return ivec2(i & 1023, i >> 10); }
vec4 ld(sampler2D s, int i) { return texelFetch(s, at(i), 0); }
int here() { ivec2 p = ivec2(gl_FragCoord.xy); return (p.y << 10) + p.x; }
vec4 silu4(vec4 x) { return x / (1.0 + exp(-x)); }
vec4 sigm4(vec4 x) { return 1.0 / (1.0 + exp(-x)); }
"#;

fn k(body: &str) -> String {
    format!("{HEAD}{body}")
}

/// Sampler names per kernel, in texture-unit order.
pub struct Kernel {
    pub name: &'static str,
    pub src: String,
    pub samplers: &'static [&'static str],
}

/// The epilogue shared by both GEMMs: `ua.w` = mode (0 store, 1 +bias, 2 R + alpha*acc, 3 silu, 4 relu(+bias)).
const EPILOGUE: &str = r#"
  vec4 v = acc;
  int mode = ua.w;
  if (mode == 1) { v += ld(B, nb); }
  else if (mode == 2) { v = ld(R, i) + uf.x * v; }
  else if (mode == 3) { v = silu4(v); }
  else if (mode == 4) { v = max(v + ld(B, nb), 0.0); }
  o = v;
"#;

pub fn kernels() -> Vec<Kernel> {
    vec![
        // Y[M][N] = X[M][K] W^T over 2-bit ternary planes: per row and 32 k one uvec4
        // (lo-plane k0..15, hi-plane k0..15, lo-plane k16..31, hi-plane k16..31), value = field - 1;
        // y = lo * (x . t_lo) + hi * (x . t_hi), four output rows per fragment. ua = (M, K, N, mode).
        Kernel {
            name: "gemm_fv",
            samplers: &["X", "W", "L", "B", "R"],
            src: k(&format!(
                r#"
uniform sampler2D X; uniform usampler2D W; uniform sampler2D L; uniform sampler2D B; uniform sampler2D R;
void main() {{
  int i = here(); if (i >= ub.w) {{ o = vec4(0.0); return; }}
  int K = ua.y; int n4 = ua.z >> 2;
  int m = i / n4; int nb = i - m * n4; int n = nb << 2;
  int kw = K >> 5; int xb = m * (K >> 2);
  vec4 al = vec4(0.0); vec4 ah = vec4(0.0);
  for (int kb = 0; kb < kw; kb++) {{
    uvec4 w0 = texelFetch(W, at(n * kw + kb), 0);
    uvec4 w1 = texelFetch(W, at((n + 1) * kw + kb), 0);
    uvec4 w2 = texelFetch(W, at((n + 2) * kw + kb), 0);
    uvec4 w3 = texelFetch(W, at((n + 3) * kw + kb), 0);
    for (int hf = 0; hf < 2; hf++) {{
      uvec4 pl = hf == 0 ? uvec4(w0.x, w1.x, w2.x, w3.x) : uvec4(w0.z, w1.z, w2.z, w3.z);
      uvec4 ph = hf == 0 ? uvec4(w0.y, w1.y, w2.y, w3.y) : uvec4(w0.w, w1.w, w2.w, w3.w);
      for (int j = 0; j < 4; j++) {{
        vec4 xv = ld(X, xb + kb * 8 + hf * 4 + j);
        uint s = uint(j * 8);
        al += xv.x * (vec4((pl >> s) & 3u) - 1.0);
        ah += xv.x * (vec4((ph >> s) & 3u) - 1.0);
        al += xv.y * (vec4((pl >> (s + 2u)) & 3u) - 1.0);
        ah += xv.y * (vec4((ph >> (s + 2u)) & 3u) - 1.0);
        al += xv.z * (vec4((pl >> (s + 4u)) & 3u) - 1.0);
        ah += xv.z * (vec4((ph >> (s + 4u)) & 3u) - 1.0);
        al += xv.w * (vec4((pl >> (s + 6u)) & 3u) - 1.0);
        ah += xv.w * (vec4((ph >> (s + 6u)) & 3u) - 1.0);
      }}
    }}
  }}
  vec4 acc = ld(L, 2 * nb) * al + ld(L, 2 * nb + 1) * ah;
  {EPILOGUE}
}}
"#
            )),
        },
        // dense f32 weights W[N][K]. ua = (M, K, N, mode)
        Kernel {
            name: "gemm_dense",
            samplers: &["X", "W", "B", "R"],
            src: k(&format!(
                r#"
uniform sampler2D X; uniform sampler2D W; uniform sampler2D B; uniform sampler2D R;
void main() {{
  int i = here(); if (i >= ub.w) {{ o = vec4(0.0); return; }}
  int K4 = ua.y >> 2; int n4 = ua.z >> 2;
  int m = i / n4; int nb = i - m * n4; int n = nb << 2;
  int xb = m * K4;
  vec4 acc = vec4(0.0);
  for (int kt = 0; kt < K4; kt++) {{
    vec4 xv = ld(X, xb + kt);
    acc.x += dot(xv, ld(W, n * K4 + kt));
    acc.y += dot(xv, ld(W, (n + 1) * K4 + kt));
    acc.z += dot(xv, ld(W, (n + 2) * K4 + kt));
    acc.w += dot(xv, ld(W, (n + 3) * K4 + kt));
  }}
  {EPILOGUE}
}}
"#
            )),
        },
        // per-row mean and 1/std over 1024 features -> (mean, rstd, 0, 0)
        Kernel {
            name: "ln_stats",
            samplers: &["X"],
            src: k(r#"
uniform sampler2D X;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int b = i << 8;
  vec4 s = vec4(0.0);
  for (int c = 0; c < 256; c++) { s += ld(X, b + c); }
  float mean = (s.x + s.y + s.z + s.w) / 1024.0;
  vec4 q = vec4(0.0);
  for (int c = 0; c < 256; c++) { vec4 d = ld(X, b + c) - mean; q += d * d; }
  o = vec4(mean, inversesqrt((q.x + q.y + q.z + q.w) / 1024.0 + 1e-5), 0.0, 0.0);
}
"#),
        },
        Kernel {
            name: "ln_apply",
            samplers: &["X", "S", "G", "Bt"],
            src: k(r#"
uniform sampler2D X; uniform sampler2D S; uniform sampler2D G; uniform sampler2D Bt;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int m = i >> 8; int c = i & 255;
  vec4 st = ld(S, m);
  o = (ld(X, i) - st.x) * st.y * ld(G, c) + ld(Bt, c);
}
"#),
        },
        // S[h][i][4 j] = scale * ((q_i + u) . k_j + (q_i + v) . P[cap-1-i+j]); masked lanes -1e30.
        // ua = (T, cap, T4, 0), uf.x = scale
        Kernel {
            name: "scores",
            samplers: &["Q", "Kt", "P", "BU", "BV"],
            src: k(r#"
uniform sampler2D Q; uniform sampler2D Kt; uniform sampler2D P; uniform sampler2D BU; uniform sampler2D BV;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int T = ua.x; int cap = ua.y; int T4 = ua.z;
  int per = T * T4; int h = i / per; int r = i - h * per; int iq = r / T4; int jt = r - iq * T4;
  int j0 = jt * 4;
  bvec4 ok = lessThan(ivec4(j0, j0 + 1, j0 + 2, j0 + 3), ivec4(T));
  int ja = min(j0, T - 1); int jb = min(j0 + 1, T - 1); int jc = min(j0 + 2, T - 1); int jd = min(j0 + 3, T - 1);
  int pa = cap - 1 - iq;
  vec4 acc = vec4(0.0);
  for (int dt = 0; dt < 32; dt++) {
    int col = h * 32 + dt;
    vec4 q = ld(Q, iq * 256 + col);
    vec4 qu = q + ld(BU, col);
    vec4 qv = q + ld(BV, col);
    acc.x += dot(qu, ld(Kt, ja * 256 + col)) + dot(qv, ld(P, (pa + ja) * 256 + col));
    acc.y += dot(qu, ld(Kt, jb * 256 + col)) + dot(qv, ld(P, (pa + jb) * 256 + col));
    acc.z += dot(qu, ld(Kt, jc * 256 + col)) + dot(qv, ld(P, (pa + jc) * 256 + col));
    acc.w += dot(qu, ld(Kt, jd * 256 + col)) + dot(qv, ld(P, (pa + jd) * 256 + col));
  }
  o = mix(vec4(-1e30), acc * uf.x, vec4(ok));
}
"#),
        },
        // per score row: (max, 1/sum exp). ua = (T, cap, T4)
        Kernel {
            name: "softmax_stats",
            samplers: &["S"],
            src: k(r#"
uniform sampler2D S;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int T4 = ua.z; int b = i * T4;
  vec4 m = vec4(-1e30);
  for (int t = 0; t < T4; t++) { m = max(m, ld(S, b + t)); }
  float mx = max(max(m.x, m.y), max(m.z, m.w));
  vec4 s = vec4(0.0);
  for (int t = 0; t < T4; t++) { s += exp(ld(S, b + t) - mx); }
  o = vec4(mx, 1.0 / (s.x + s.y + s.z + s.w), 0.0, 0.0);
}
"#),
        },
        Kernel {
            name: "softmax_apply",
            samplers: &["S", "St"],
            src: k(r#"
uniform sampler2D S; uniform sampler2D St;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  vec4 st = ld(St, i / ua.z);
  o = exp(ld(S, i) - st.x) * st.y;
}
"#),
        },
        // C[i][h*128 + 4c] = sum_j p[h][i][j] V[j][..]. ua = (T, cap, T4)
        Kernel {
            name: "attn_av",
            samplers: &["Pr", "V"],
            src: k(r#"
uniform sampler2D Pr; uniform sampler2D V;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int T = ua.x; int T4 = ua.z;
  int iq = i >> 8; int c = i & 255; int h = c >> 5;
  int pb = (h * T + iq) * T4;
  vec4 acc = vec4(0.0);
  for (int jt = 0; jt < T4; jt++) {
    vec4 p = ld(Pr, pb + jt);
    int j = jt * 4;
    acc += p.x * ld(V, min(j, T - 1) * 256 + c);
    acc += p.y * ld(V, min(j + 1, T - 1) * 256 + c);
    acc += p.z * ld(V, min(j + 2, T - 1) * 256 + c);
    acc += p.w * ld(V, min(j + 3, T - 1) * 256 + c);
  }
  o = acc;
}
"#),
        },
        Kernel {
            name: "glu",
            samplers: &["A"],
            src: k(r#"
uniform sampler2D A;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int t = i >> 8; int c = i & 255;
  o = ld(A, t * 512 + c) * sigm4(ld(A, t * 512 + 256 + c));
}
"#),
        },
        // depthwise conv over time k=9 pad 4, folded batch norm, SiLU. ua.x = T. Weights [tap][1024].
        Kernel {
            name: "dwconv",
            samplers: &["G", "Wd", "Sc", "Sh"],
            src: k(r#"
uniform sampler2D G; uniform sampler2D Wd; uniform sampler2D Sc; uniform sampler2D Sh;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int T = ua.x; int t = i >> 8; int c = i & 255;
  vec4 acc = vec4(0.0);
  for (int k = 0; k < 9; k++) {
    int s = t + k - 4;
    if (s >= 0 && s < T) { acc += ld(Wd, k * 256 + c) * ld(G, s * 256 + c); }
  }
  o = silu4(acc * ld(Sc, c) + ld(Sh, c));
}
"#),
        },
        // subsampling front: conv 1->256 3x3/2 (+b, ReLU) under depthwise 3x3/2 (+b). Out [H2][32][256].
        // ua = (mel frames, H1, H2). Weights [tap][256].
        Kernel {
            name: "conv0_dw1",
            samplers: &["Mel", "W0", "B0", "W1", "B1"],
            src: k(r#"
uniform sampler2D Mel; uniform sampler2D W0; uniform sampler2D B0; uniform sampler2D W1; uniform sampler2D B1;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int TM = ua.x; int H1 = ua.y;
  int c = i & 63; int x = (i >> 6) & 31; int y = i >> 11;
  vec4 acc = ld(B1, c);
  vec4 b0 = ld(B0, c);
  for (int ky = 0; ky < 3; ky++) {
    int pp = 2 * y + ky - 1;
    if (pp < 0 || pp >= H1) continue;
    for (int kx = 0; kx < 3; kx++) {
      int q = 2 * x + kx - 1;
      if (q < 0 || q >= 64) continue;
      vec4 v = b0;
      for (int a = 0; a < 3; a++) {
        int r = 2 * pp + a - 1;
        if (r < 0 || r >= TM) continue;
        for (int b = 0; b < 3; b++) {
          int s = 2 * q + b - 1;
          if (s < 0 || s >= 128) continue;
          v += ld(W0, (a * 3 + b) * 64 + c) * ld(Mel, r * 32 + (s >> 2))[s & 3];
        }
      }
      acc += ld(W1, (ky * 3 + kx) * 64 + c) * max(v, 0.0);
    }
  }
  o = acc;
}
"#),
        },
        // depthwise 3x3/2 pad 1 over [H][W][256] (+b). ua = (H, W, Ho, Wo)
        Kernel {
            name: "dw_s2",
            samplers: &["I", "Wd", "Bd"],
            src: k(r#"
uniform sampler2D I; uniform sampler2D Wd; uniform sampler2D Bd;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int H = ua.x; int W = ua.y; int Wo = ua.w;
  int c = i & 63; int pos = i >> 6; int x = pos % Wo; int y = pos / Wo;
  vec4 acc = ld(Bd, c);
  for (int ky = 0; ky < 3; ky++) {
    int iy = 2 * y + ky - 1;
    if (iy < 0 || iy >= H) continue;
    for (int kx = 0; kx < 3; kx++) {
      int ix = 2 * x + kx - 1;
      if (ix < 0 || ix >= W) continue;
      acc += ld(Wd, (ky * 3 + kx) * 64 + c) * ld(I, (iy * W + ix) * 64 + c);
    }
  }
  o = acc;
}
"#),
        },
        // [T][16 freq][256 ch] -> [T][ch * 16 + freq]
        Kernel {
            name: "flatten",
            samplers: &["I"],
            src: k(r#"
uniform sampler2D I;
void main() {
  int i = here(); if (i >= ub.w) { o = vec4(0.0); return; }
  int t = i >> 10; int r = i & 1023; int c = r >> 2; int f0 = (r & 3) * 4;
  int cc = c >> 2; int lane = c & 3;
  o = vec4(ld(I, (t * 16 + f0) * 64 + cc)[lane], ld(I, (t * 16 + f0 + 1) * 64 + cc)[lane],
           ld(I, (t * 16 + f0 + 2) * 64 + cc)[lane], ld(I, (t * 16 + f0 + 3) * 64 + cc)[lane]);
}
"#),
        },
    ]
}
