import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

export const root = fileURLToPath(new URL('../../', import.meta.url));
export const directory = fileURLToPath(new URL('./', import.meta.url));
export const hash = (b) => createHash('sha256').update(b).digest('hex');
export const readJSON = async (p) => JSON.parse(await readFile(path.join(root, p)));

export function validateSelection(row, manifest, config) {
  const ids = row.selections.map((c) => c.image_id).sort();

  if (JSON.stringify(ids) !== JSON.stringify([...config.image_ids].sort()))
    throw Error('Expected one choice for each of 14 images');

  let bytes = 0,
    ssim = 0;

  for (const s of row.selections) {
    const c = manifest.find((c) => c.image_id === s.image_id && c.version === s.version);

    if (
      !c ||
      ['bytes', 'ssim', 'path', 'sha256', 'width', 'height'].some((k) => c[k] !== s[k])
    )
      throw Error('Selection differs from manifest');

    if (c.ssim < (config.important_thresholds[c.image_id] ?? config.ordinary_threshold))
      throw Error('Quality floor violation');

    bytes += c.bytes;
    ssim += c.ssim;
  }

  if (
    bytes !== row.total_bytes ||
    bytes > row.budget_bytes ||
    Math.abs(ssim / 14 - row.average_ssim) > 1e-10
  )
    throw Error('Invalid saved bytes/budget/objective');

  if (row.budget_bytes !== (row.scenario_mbps * 1000000 * 2.5) / 8 || !row.feasible)
    throw Error('Invalid scenario');

  if (row.method === 'milp' && (row.status !== 'optimal' || row.optimality_gap !== 0))
    throw Error('Unproven MILP');
}

export async function loadInputs() {
  const manifest = await readJSON('revisions/user-selected-v1/manifest.json');
  const config = await readJSON('revisions/user-selected-v1/config.json');
  const frozen = await readJSON('revisions/user-selected-v1/freeze.json');

  if (
    hash(await readFile(path.join(root, 'revisions/user-selected-v1/manifest.json'))) !==
    frozen.manifest_sha256
  )
    throw Error('Manifest hash differs');

  for (const c of manifest) {
    const b = await readFile(path.join(root, c.path));

    if (b.length !== c.bytes || hash(b) !== c.sha256)
      throw Error(`Candidate integrity: ${c.path}`);
  }

  const { runs } = await readJSON('revisions/part1-milp-ga-v1/results/raw.json');

  if (runs.length !== 180) throw Error('Expected180 original runs');

  for (const speed of [4, 8, 12])
    for (const method of ['milp', 'ga'])
      for (let rep = 0; rep < 30; rep++) {
        const rows = runs.filter(
          (r) => r.scenario_mbps === speed && r.method === method && r.repetition === rep,
        );

        if (rows.length !== 1) throw Error('Missing/duplicate trial');

        validateSelection(rows[0], manifest, config);
      }

  return {
    manifest,
    config,
    runs,
  };
}

const css = `*{box-sizing:border-box}body{margin:0;padding:24px;font:18px system-ui,sans-serif;color:#182a2a;background:#f7f8f5}h1{font-size:28px;margin:0 0 8px}p{margin:8px 0 22px}#gallery{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:14px}figure{margin:0;padding:8px;background:white;border:1px solid #cad2ce}.image-frame{height:190px;width:100%;display:flex;align-items:center;justify-content:center;background:#eee}img{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain}figcaption{font-size:16px;margin-top:7px}footer{font-size:15px;margin-top:22px}a{color:#155944}select,button{font:inherit;padding:9px}label{display:block;margin:16px 0}@media(max-width:800px){#gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}`;

function gallery(row) {
  const cards = [...row.selections]
    .sort((a, b) => a.image_id.localeCompare(b.image_id))
    .map(
      (c) =>
        `<figure data-image-id="${c.image_id}"><div class="image-frame"><img id="image-${c.image_id}" src="/images/${c.image_id}-${c.version}.webp" alt="DIV2K ${c.image_id}" width="${c.width}" height="${c.height}" loading="eager" decoding="async" fetchpriority="auto"></div><figcaption>Image ${c.image_id}</figcaption></figure>`,
    )
    .join('');

  return `<!doctype html><html lang="en" data-ready="true"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><link rel="stylesheet" href="/style.css"><title>Part1 visible gallery</title><body data-feasible="true"><h1>Fourteen images, one shared connection</h1><p>Controlled local browser test · each image is fetched once.</p><main id="gallery">${cards}</main><footer>Recorded selections · no optimization during loading · browser readiness is measured separately from estimated payload transfer time.</footer></body></html>`;
}

export async function createServer(inputs) {
  inputs ??= await loadInputs();
  const images = new Map(
    inputs.manifest.map((c) => [`/images/${c.image_id}-${c.version}.webp`, c]),
  );
  const logs = [];
  const server = http.createServer(async (req, res) => {
    const u = new URL(req.url, 'http://localhost');
    let body,
      type = 'text/html; charset=utf-8',
      status = 200;

    try {
      if (u.pathname === '/style.css') {
        body = css;
        type = 'text/css; charset=utf-8';
      } else if (images.has(u.pathname)) {
        body = await readFile(path.join(root, images.get(u.pathname).path));
        type = 'image/webp';
      } else if (u.pathname === '/gallery') {
        const r = inputs.runs.find(
          (r) =>
            r.method === u.searchParams.get('method') &&
            r.scenario_mbps === Number(u.searchParams.get('mbps')) &&
            r.repetition === Number(u.searchParams.get('rep')),
        );

        if (!r) {
          status = 400;
          body = 'Unknown recorded trial';
        } else body = gallery(r);
      } else if (u.pathname === '/')
        body = `<!doctype html><html lang="en"><meta charset="utf-8"><link rel="icon" href="data:,"><link rel="stylesheet" href="/style.css"><title>Part1 browser lab</title><h1>Part1 browser lab</h1><p>Unchanged measured images. Manual viewing is not throttled. Use run.mjs for controlled measurements.</p><form action="/gallery"><label>Recorded method <select name="method"><option value="milp">MILP</option><option value="ga">Original GA</option></select></label><label>Budget scenario <select name="mbps"><option>4</option><option>8</option><option>12</option></select> Mbps ×2.5 seconds</label><label>Recorded trial <input name="rep" type="number" min="0" max="29" value="0"></label><button>Open gallery</button></form></html>`;
      else {
        status = 404;
        body = 'Not found';
      }
    } catch (e) {
      status = 500;
      body = 'File read failed';
    }

    const buffer = Buffer.isBuffer(body) ? body : Buffer.from(body);
    logs.push({
      at_utc: new Date().toISOString(),
      method: req.method,
      url: req.url,
      status,
      body_bytes: buffer.length,
    });
    res.writeHead(status, {
      'Content-Type': type,
      'Content-Length': buffer.length,
      'Cache-Control': 'no-store',
    });
    res.end(buffer);
  });

  return {
    server,
    logs,
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { server } = await createServer();
  const port = Number(process.env.PORT ?? 8096);
  server.listen(port, '127.0.0.1', () =>
    console.log(`Local gallery: http://127.0.0.1:${port} (manual viewing is unthrottled)`),
  );
}
