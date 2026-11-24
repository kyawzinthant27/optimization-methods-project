export function makePlan(runs) {
  const plan = [];

  for (let rep = 0; rep < 30; rep++) {
    const speeds = [4, 8, 12],
      order = speeds.slice(rep % 3).concat(speeds.slice(0, rep % 3));

    for (const speed of order)
      for (const method of rep % 2 ? ['ga', 'milp'] : ['milp', 'ga']) {
        const row = runs.find(
          (r) => r.scenario_mbps === speed && r.method === method && r.repetition === rep,
        );
        plan.push({
          run_id: `run-${String(plan.length + 1).padStart(3, '0')}`,
          method,
          scenario_mbps: speed,
          repetition: rep,
          seed: row.seed,
        });
      }
  }

  return plan;
}

export function validateRun(s, network, row, frames) {
  if (!s) return ['Missing snapshot'];

  const failures = [];
  const expected = row.selections
    .map((c) => `/images/${c.image_id}-${c.version}.webp`)
    .sort();
  const same = (a) => JSON.stringify([...a].sort()) === JSON.stringify(expected);
  const resources = s.resources.filter((r) => r.path.startsWith('/images/'));

  for (const [name, paths] of Object.entries({
    requests: network.filter((r) => r.path.startsWith('/images/')).map((r) => r.path),
    resources: resources.map((r) => r.path),
    loads: s.images.map((r) => r.path),
    DOM: s.gallery.map((r) => r.path),
  }))
    if (!same(paths)) failures.push(`Expected exactly14 selected ${name}`);

  if (network.some((r) => !r.finished || r.failed || r.status !== 200))
    failures.push('HTTP request failed/incomplete');

  if (
    network.some(
      (r) =>
        r.from_disk_cache ||
        r.from_service_worker ||
        r.from_prefetch_cache ||
        r.served_from_cache,
    )
  )
    failures.push('Cache detected');

  if (
    JSON.stringify(
      network
        .filter((r) => !r.path.startsWith('/images/'))
        .map((r) => r.path)
        .sort(),
    ) !== JSON.stringify(['/gallery', '/style.css'])
  )
    failures.push('Unexpected non-image request');

  for (const r of resources) {
    const c = row.selections.find(
      (c) => `/images/${c.image_id}-${c.version}.webp` === r.path,
    );

    if (!c || c.bytes !== r.encoded_body_bytes || r.transfer_bytes <= 0)
      failures.push(`Body mismatch ${r.path}`);
  }

  for (const i of s.gallery) {
    const c = row.selections.find(
      (c) => `/images/${c.image_id}-${c.version}.webp` === i.path,
    );

    if (
      !c ||
      !i.complete ||
      i.natural_width !== c.width ||
      i.natural_height !== c.height ||
      i.width_attribute !== c.width ||
      i.height_attribute !== c.height ||
      i.loading !== 'eager' ||
      i.decoding !== 'async' ||
      i.fetch_priority !== 'auto' ||
      i.srcset
    )
      failures.push('Incorrect dimensions/loading behavior');
  }

  if (s.images.some((i) => i.decode_ready_ms === null || i.decode_error))
    failures.push('Decode failed/incomplete');

  if (
    s.frames.length !== 14 ||
    s.frames.some(
      (f) => f.x < 0 || f.y < 0 || f.x + f.width > 1440 || f.y + f.height > 1100,
    )
  )
    failures.push('Gallery not entirely visible');

  if (frames && JSON.stringify(frames) !== JSON.stringify(s.frames))
    failures.push('Layout changed');

  if (s.fcp_ms === null || !s.lcp_entries.length) failures.push('Paint metrics missing');

  if (
    s.visibility !== 'visible' ||
    s.visibility_changes.some((v) => v.state !== 'visible') ||
    s.scroll_x ||
    s.scroll_y
  )
    failures.push('Hidden/scrolled page');

  if (s.page_ready !== 'true' || s.feasible !== 'true') failures.push('Page not ready');

  if (resources.reduce((a, r) => a + r.encoded_body_bytes, 0) !== row.total_bytes)
    failures.push('Payload sum differs');

  return [...new Set([...failures, ...s.errors])];
}

export function metrics(snapshot) {
  if (!snapshot) return {};

  const images = snapshot.resources.filter((r) => r.path.startsWith('/images/'));
  const first = images.length ? Math.min(...images.map((r) => r.start_ms)) : null;
  const last = images.length ? Math.max(...images.map((r) => r.response_end_ms)) : null;
  const ready = snapshot.images.filter((i) => i.decode_ready_ms !== null);
  const decoded = ready.length ? Math.max(...ready.map((i) => i.decode_ready_ms)) : null;
  const lcp = snapshot.lcp_entries.at(-1);

  return {
    image_payload_bytes: images.reduce((sum, r) => sum + r.encoded_body_bytes, 0),
    image_transfer_bytes: images.reduce((sum, r) => sum + r.transfer_bytes, 0),
    first_image_start_ms: first,
    image_requests_complete_ms: last,
    image_request_span_ms: first === null ? null : last - first,
    image_load_events_complete_ms: snapshot.images.length
      ? Math.max(...snapshot.images.map((i) => i.load_event_ms))
      : null,
    image_decode_ready_ms: decoded,
    image_readiness_span_ms: first === null || decoded === null ? null : decoded - first,
    decode_after_last_response_ms:
      last === null || decoded === null ? null : decoded - last,
    fcp_ms: snapshot.fcp_ms,
    lcp_ms: lcp?.start_time_ms ?? null,
    lcp_tag: lcp?.tag ?? null,
    lcp_id: lcp?.id ?? null,
    lcp_text: lcp?.text ?? null,
    lcp_url: lcp?.url ?? null,
    observation_end_ms: snapshot.observation_end_ms,
  };
}
