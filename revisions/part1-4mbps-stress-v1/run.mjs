import { chromium } from 'playwright';
import { readFile, writeFile, mkdir, appendFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  root,
  directory as original,
  loadInputs,
  createServer,
  hash,
} from '../part1-browser-benchmark-v1/server.mjs';

const directory = fileURLToPath(new URL('./', import.meta.url));
import { validateRun, metrics } from '../part1-browser-benchmark-v1/protocol.mjs';

const pilot = false,
  output = path.join(directory, 'results');
const sourceSettings = JSON.parse(await readFile(path.join(original, 'plan.json')));
const settings = {
  ...sourceSettings,
  repetitions: 12,
  networks_mbps: [4],
  revision: 'part1-4mbps-stress-v1',
  question:
    'How does tightening the byte cap at 4 Mbps affect the feasible boundary, quality, and actual browser readiness?',
  selections:
    'Exact enumeration winners on a fixed dataset; one frozen selection per distinct case.',
  order:
    'Twelve paired rounds, rotated case order with alternating direction in blocks of case-count rounds.',
  pilot: 'No instrumentation pilots in this follow-up.',
  statistics:
    'Mean SSIM, byte totals, median and quartiles, minimum observed time, paired differences to reference, all failures.',
  target: '2.5 seconds is the reference payload allowance. It is not a browser deadline.',
};
const { browser_cases } = JSON.parse(await readFile(path.join(directory, 'cases.json')));
const inputs = {
  ...(await loadInputs()),
  runs: browser_cases,
};
const plan = [];

for (let round = 0; round < settings.repetitions; round++) {
  const cases =
    round >= browser_cases.length ? [...browser_cases].reverse() : browser_cases;
  const shift = round % cases.length;

  for (const row of [...cases.slice(shift), ...cases.slice(0, shift)]) {
    plan.push({
      run_id: `run-${String(plan.length + 1).padStart(3, '0')}`,
      method: row.method,
      scenario_mbps: 4,
      repetition: row.repetition,
      round,
    });
  }
}

await mkdir(output, {
  recursive: true,
});
await writeFile(path.join(output, 'runs.jsonl'), '', {
  flag: 'wx',
});
const save = (name, obj) =>
  writeFile(path.join(output, name), JSON.stringify(obj, null, 2) + '\n');
const files = [
  ...settings.sources,
  'package-lock.json',
  'revisions/part1-milp-ga-v1/plan.json',
  'revisions/part1-milp-ga-v1/ga.py',
  'image_quality/optimize.py',
  ...inputs.manifest.map((c) => c.path),
  ...['plan.json', 'server.mjs', 'run.mjs', 'protocol.mjs', 'instrument.js'].map((f) =>
    path.relative(root, path.join(original, f)),
  ),
  ...['README.md', 'plan.py', 'cases.json', 'run.mjs'].map((f) =>
    path.relative(root, path.join(directory, f)),
  ),
  'revisions/part1-milp-budget-sensitivity-v1/results/summary.csv',
];
const fingerprint = async () =>
  Object.fromEntries(
    await Promise.all(
      files.map(async (f) => [f, hash(await readFile(path.join(root, f)))]),
    ),
  );
const before = await fingerprint();
await save('plan.json', {
  saved_at_utc: new Date().toISOString(),
  pilot,
  settings,
  plan,
  source_sha256: before,
});
const { server, logs } = await createServer(inputs);
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const chromePath = process.env.CHROME_PATH || '/usr/bin/google-chrome';
const browser = await chromium.launch({
  executablePath: chromePath,
  headless: true,
});
const browserCdp = await browser.newBrowserCDPSession();
const metadata = {
  started_at_utc: new Date().toISOString(),
  pilot,
  settings,
  source_sha256_before: before,
  environment: {
    node: process.version,
    playwright: JSON.parse(
      await readFile(path.join(root, 'node_modules/playwright/package.json')),
    ).version,
    browser: await browserCdp.send('Browser.getVersion'),
    chrome_path: chromePath,
    headless: true,
    os_type: os.type(),
    os_release: os.release(),
    os_version: os.version(),
    arch: os.arch(),
    cpu_model: os.cpus()[0]?.model,
    logical_cpus: os.cpus().length,
    total_memory_bytes: os.totalmem(),
    initial_load_average: os.loadavg(),
    gpu: await browserCdp.send('SystemInfo.getInfo').catch((e) => ({
      error: e.message,
    })),
  },
  server: {
    base_url: base,
    implementation:
      'Same Node HTTP/1.1 process throughout, readFile per image, no-store. No server delay or compression.',
    cache_scope: settings.cache,
  },
  definitions: {
    common_origin:
      'performance.timeOrigin (navigation start). All browser times in milliseconds.',
    image_requests_complete_ms:
      'Maximum responseEnd across the 14 selected images, relative to navigation.',
    image_request_span_ms:
      'Last image responseEnd minus earliest image startTime. Includes browser queueing and latency.',
    image_decode_ready_ms:
      'Latest decode() resolution for all14 images, relative to navigation. decode() called on each load event. Not proof all pixels painted.',
    image_readiness_span_ms: 'All-image decode readiness minus earliest image fetch start.',
    fcp_ms: 'first-contentful-paint startTime from navigation origin.',
    lcp_ms: 'Last largest-contentful-paint entry at cutoff, with element identity saved.',
    observation_cutoff:
      '1000ms after all decode promises settle, or after timeout/error; no scrolling or interaction.',
    image_payload_bytes: 'Sum encodedBodySize; must equal the selected files.',
    image_transfer_bytes:
      'Sum ResourceTiming.transferSize; includes browser-reported headers, not physical-link bytes.',
    latency: 'CDP minimum request-to-response-header delay, not measured physical RTT.',
    theoretical_transfer_ms:
      'Selected bytes*8/(Mbps*1000000)*1000; estimate excludes latency, competing traffic, non-image bytes and decode.',
    solver_runtime_ms_precomputed:
      'No solver timing supplied for precomputed exact-enumeration selections.',
    target:
      'Modeled2.5s allowance; navigation/readiness span attainment counted separately.',
  },
};
await save('metadata.json', metadata);
const runs = [];
let baselineFrames;

try {
  for (const item of plan) {
    const row = inputs.runs.find(
      (r) =>
        r.method === item.method &&
        r.scenario_mbps === item.scenario_mbps &&
        r.repetition === item.repetition,
    );
    const context = await browser.newContext({
      viewport: settings.viewport,
      deviceScaleFactor: 1,
      serviceWorkers: 'block',
      locale: 'en-US',
      timezoneId: 'UTC',
      colorScheme: 'light',
      reducedMotion: 'no-preference',
    });
    await context.addInitScript({
      path: path.join(original, 'instrument.js'),
    });
    const page = await context.newPage(),
      cdp = await context.newCDPSession(page),
      network = new Map(),
      errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    cdp.on('Network.requestWillBeSent', (e) => {
      if (!e.request.url.startsWith(base)) {
        if (/^https?:/.test(e.request.url))
          errors.push(`Unexpected external request: ${e.request.url}`);

        return;
      }

      network.set(e.requestId, {
        request_id: e.requestId,
        path: new URL(e.request.url).pathname,
        request_timestamp: e.timestamp,
        wall_time_seconds: e.wallTime,
        resource_type: e.type,
        initial_priority: e.request.initialPriority,
        status: null,
        finished: false,
        failed: null,
        served_from_cache: false,
      });
    });
    cdp.on('Network.responseReceived', (e) => {
      const r = network.get(e.requestId);

      if (r)
        Object.assign(r, {
          status: e.response.status,
          protocol: e.response.protocol,
          response_timestamp: e.timestamp,
          response_timing: e.response.timing,
          from_disk_cache: !!e.response.fromDiskCache,
          from_service_worker: !!e.response.fromServiceWorker,
          from_prefetch_cache: !!e.response.fromPrefetchCache,
          headers: e.response.headers,
        });
    });
    cdp.on('Network.requestServedFromCache', (e) => {
      const r = network.get(e.requestId);

      if (r) r.served_from_cache = true;
    });
    cdp.on('Network.loadingFinished', (e) => {
      const r = network.get(e.requestId);

      if (r)
        Object.assign(r, {
          finished: true,
          finish_timestamp: e.timestamp,
          cdp_encoded_transfer_bytes: e.encodedDataLength,
        });
    });
    cdp.on('Network.loadingFailed', (e) => {
      const r = network.get(e.requestId);

      if (r) r.failed = e.errorText;
    });
    const commands = {
      'Network.setCacheDisabled': {
        cacheDisabled: true,
      },
      'Network.clearBrowserCache': {},
      'Network.setBypassServiceWorker': {
        bypass: true,
      },
      'Emulation.setCPUThrottlingRate': {
        rate: 1,
      },
      'Network.emulateNetworkConditions': {
        offline: false,
        latency: settings.latency_ms,
        downloadThroughput: (item.scenario_mbps * 1000000) / 8,
        uploadThroughput: (item.scenario_mbps * 1000000) / 8,
        connectionType: 'cellular4g',
      },
    };
    const started = new Date().toISOString(),
      hostLoad = os.loadavg(),
      logStart = logs.length;
    let snapshot = null;

    try {
      await cdp.send('Network.enable');

      for (const [method, params] of Object.entries(commands))
        await cdp.send(method, params);

      await page.goto(
        `${base}/gallery?method=${item.method}&mbps=${item.scenario_mbps}&rep=${item.repetition}`,
        {
          waitUntil: 'domcontentloaded',
          timeout: settings.timeout_ms,
        },
      );
      await page.waitForFunction(() => window.__imageBenchmark?.done(), null, {
        timeout: settings.timeout_ms,
      });
    } catch (e) {
      errors.push(e.message);
    }

    try {
      await page.waitForTimeout(settings.post_readiness_observation_ms);
      snapshot = await page.evaluate(() => window.__imageBenchmark?.snapshot() ?? null);
    } catch (e) {
      errors.push(`Snapshot failed: ${e.message}`);
    }

    const requests = [...network.values()];
    errors.push(...validateRun(snapshot, requests, row, baselineFrames));
    const serverRequests = logs.slice(logStart),
      expected = row.selections
        .map((c) => `/images/${c.image_id}-${c.version}.webp`)
        .sort();

    if (
      JSON.stringify(
        serverRequests
          .filter((r) => r.url.startsWith('/images/'))
          .map((r) => r.url)
          .sort(),
      ) !== JSON.stringify(expected)
    )
      errors.push('Server did not receive exactly14 selected requests');

    if (
      serverRequests
        .filter((r) => r.url.startsWith('/images/'))
        .reduce((a, r) => a + r.body_bytes, 0) !== row.total_bytes
    )
      errors.push('Server byte sum mismatch');

    const success = errors.length === 0;

    if (success && !baselineFrames) baselineFrames = snapshot.frames;

    const measured = metrics(snapshot);
    const result = {
      ...item,
      success,
      failure_reasons: [...new Set(errors)],
      started_at_utc: started,
      finished_at_utc: new Date().toISOString(),
      host_load_average: hostLoad,
      cdp_commands: commands,
      budget_bytes: row.budget_bytes,
      selected_bytes: row.total_bytes,
      average_ssim: row.average_ssim,
      minimum_ssim: row.minimum_ssim,
      selections: row.selections.map((c) => ({
        image_id: c.image_id,
        version: c.version,
        bytes: c.bytes,
        ssim: c.ssim,
        sha256: c.sha256,
      })),
      solver_runtime_ms_precomputed: null,
      theoretical_transfer_ms:
        ((row.total_bytes * 8) / (item.scenario_mbps * 1000000)) * 1000,
      metrics: measured,
      snapshot,
      requests,
      server_requests: serverRequests,
      target_navigation_readiness_met: success
        ? measured.image_decode_ready_ms <= 2500
        : null,
      target_batch_readiness_met: success ? measured.image_readiness_span_ms <= 2500 : null,
    };
    runs.push(result);
    await appendFile(path.join(output, 'runs.jsonl'), JSON.stringify(result) + '\n');

    if (pilot)
      await page.screenshot({
        path: path.join(output, `${item.run_id}.png`),
        fullPage: true,
      });

    console.log(
      `${runs.length}/${plan.length} ${item.scenario_mbps}Mbps ${item.method} rep=${item.repetition} ${success ? 'PASS' : 'FAIL'} ready=${measured.image_decode_ready_ms?.toFixed(1)}ms ${errors.join('; ')}`,
    );
    await context.close();
  }
} finally {
  metadata.finished_at_utc = new Date().toISOString();
  metadata.final_load_average = os.loadavg();
  metadata.source_sha256_after = await fingerprint();
  metadata.inputs_unchanged =
    JSON.stringify(before) === JSON.stringify(metadata.source_sha256_after);
  await save('metadata.json', metadata);
  await save('raw.json', {
    metadata,
    runs,
  });
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}

if (!metadata.inputs_unchanged) throw Error('Inputs changed during benchmark');

console.log(
  `Saved ${runs.length} attempts, ${runs.filter((r) => !r.success).length} failures. ${output}`,
);
