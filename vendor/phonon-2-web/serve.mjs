#!/usr/bin/env node
// Static server for web/ with the cross-origin isolation headers that WebAssembly threads need
// (SharedArrayBuffer): COOP same-origin + COEP require-corp.
//
//   node serve.mjs [port]                         http://localhost:8787
//   PHONON_MODEL=/path/to/phonon-2.bps.tar.zst node serve.mjs
//       also serves that file at /model/phonon-2.bps.tar.zst; open /?model=/model/phonon-2.bps.tar.zst to skip the
//       Hugging Face download while developing. /test.html checks every backend against testdata/ref.
//   ROOT=dist/space node serve.mjs                 serve the assembled Space instead of web/

import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));
const root = process.env.ROOT ? join(process.cwd(), process.env.ROOT) : join(here, 'web');
const testdata = join(here, 'testdata');
const port = Number(process.argv[2] || process.env.PORT || 8787);
const model = process.env.PHONON_MODEL;
const types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.wasm': 'application/wasm', '.json': 'application/json', '.svg': 'image/svg+xml',
  '.zst': 'application/octet-stream', '.wav': 'audio/wav', '.ico': 'image/x-icon',
};

createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  // /plain/... serves the same files without the isolation headers (exercises the single-threaded fallback)
  const plain = url.pathname.startsWith('/plain/');
  if (plain) url.pathname = url.pathname.slice(6);
  let file;
  if (model && url.pathname === '/model/phonon-2.bps.tar.zst') file = model;
  else if (url.pathname.startsWith('/testdata/')) {
    file = join(testdata, normalize(decodeURIComponent(url.pathname.slice(10))).replace(/^(\.\.[/\\])+/, ''));
    if (!file.startsWith(testdata)) file = null;
  } else {
    const p = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, '');
    file = join(root, p.endsWith('/') ? `${p}index.html` : p);
    if (!file.startsWith(root)) file = null;
  }
  let st;
  try { st = file && statSync(file); } catch { st = null; }
  if (!st || !st.isFile()) {
    res.writeHead(404, { 'content-type': 'text/plain' });
    return res.end('not found');
  }
  res.writeHead(200, {
    'content-type': types[extname(file)] || 'application/octet-stream',
    'content-length': st.size,
    ...(plain ? {} : {
      'cross-origin-opener-policy': 'same-origin',
      'cross-origin-embedder-policy': 'require-corp',
      'cross-origin-resource-policy': 'same-origin',
    }),
    'cache-control': 'no-cache',
  });
  if (req.method === 'HEAD') return res.end();
  createReadStream(file).pipe(res);
}).listen(port, () => {
  console.log(`Phonon-2 web: http://localhost:${port}/` + (model ? `?model=/model/phonon-2.bps.tar.zst` : ''));
});
