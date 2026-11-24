import csv, json, math, statistics, sys, html
from pathlib import Path
from collections import Counter
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / 'results'
data = json.loads((OUT / 'raw.json').read_text())
runs = data['runs']
meta = data['metadata']

assert not meta['pilot'] and len(runs) == 180 and meta['inputs_unchanged']


def quantile(values, p):
    x = sorted(values)
    if not x:
        return None

    a = (len(x) - 1) * p
    i = math.floor(a)
    j = math.ceil(a)

    return x[i] + (x[j] - x[i]) * (a - i)


def stats(v):
    return dict(
        n=len(v),
        minimum=min(v) if v else None,
        q1=quantile(v, 0.25),
        median=quantile(v, 0.5),
        q3=quantile(v, 0.75),
        maximum=max(v) if v else None,
        mean=statistics.mean(v) if v else None
    )


METRICS = [
    'image_requests_complete_ms',
    'image_request_span_ms',
    'image_decode_ready_ms',
    'image_readiness_span_ms',
    'fcp_ms',
    'lcp_ms'
]
summary = []
paired = []

for speed in (4, 8, 12):
    for method in ('milp', 'ga'):
        group = [r for r in runs if r['scenario_mbps'] == speed and r['method'] == method]
        assert len(group) == 30
        good = [r for r in group if r['success']]
        summary.append(dict(
            scenario_mbps=speed,
            method=method,
            attempts=30,
            successes=len(good),
            failures=30 - len(good),
            selected_bytes=stats([r['selected_bytes'] for r in group]),
            average_ssim=stats([r['average_ssim'] for r in group]),
            recorded_solver_runtime_ms=stats([r['solver_runtime_ms_precomputed'] for r in group]),
            estimated_payload_transfer_ms=stats([r['theoretical_transfer_ms'] for r in group]),
            metrics={m: stats([r['metrics'][m] for r in good]) for m in METRICS},
            navigation_ready_within_2500_ms=sum((
                r['target_navigation_readiness_met'] is True
                for r in group
            )),
            batch_ready_within_2500_ms=sum((
                r['target_batch_readiness_met'] is True
                for r in group
            )),
            lcp_elements=dict(Counter((
                f"{r['metrics']['lcp_tag']}#{r['metrics']['lcp_id']}"
                for r in good
            )))
        ))

    for rep in range(30):
        m = next((
            r
            for r in runs
            if (r['scenario_mbps'], r['method'], r['repetition']) == (speed, 'milp', rep)
        ))
        g = next((
            r
            for r in runs
            if (r['scenario_mbps'], r['method'], r['repetition']) == (speed, 'ga', rep)
        ))
        sig = lambda r: sorted(((c['image_id'], c['version']) for c in r['selections']))
        paired.append(dict(
            scenario_mbps=speed,
            repetition=rep,
            ga_seed=g['seed'],
            same_selection=sig(g) == sig(m),
            quality_match=abs(g['average_ssim'] - m['average_ssim']) <= 1e-10,
            ssim_ga_minus_milp=g['average_ssim'] - m['average_ssim'],
            bytes_ga_minus_milp=g['selected_bytes'] - m['selected_bytes'],
            both_success=m['success'] and g['success'],
            readiness_ga_minus_milp_ms=g['metrics']['image_decode_ready_ms'] - m['metrics']['image_decode_ready_ms'] if m['success'] and g['success'] else None
        ))

paired_summary = []

for speed in (4, 8, 12):
    g = [p for p in paired if p['scenario_mbps'] == speed]
    paired_summary.append(dict(
        scenario_mbps=speed,
        quality_matches=sum((p['quality_match'] for p in g)),
        same_selections=sum((p['same_selection'] for p in g)),
        both_success=sum((p['both_success'] for p in g)),
        readiness_ga_minus_milp_ms=stats([
            p['readiness_ga_minus_milp_ms']
            for p in g
            if p['both_success']
        ])
    ))

(OUT / 'summary.json').write_text(json.dumps(
    {'groups': summary, 'paired': paired_summary},
    indent=2
) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


flat = []

for r in runs:
    flat.append({k: r.get(k) for k in [
        'run_id',
        'method',
        'scenario_mbps',
        'repetition',
        'seed',
        'success',
        'budget_bytes',
        'selected_bytes',
        'average_ssim',
        'solver_runtime_ms_precomputed',
        'theoretical_transfer_ms',
        'target_navigation_readiness_met',
        'target_batch_readiness_met'
    ]} | {'failure_reasons': '; '.join(r['failure_reasons'])} | r['metrics'])

write_csv('raw.csv', flat)
write_csv('paired.csv', paired)
compact = []

for s in summary:
    row = {k: s[k] for k in [
        'scenario_mbps',
        'method',
        'attempts',
        'successes',
        'failures',
        'navigation_ready_within_2500_ms',
        'batch_ready_within_2500_ms'
    ]}
    for m in METRICS:
        for q in ('q1', 'median', 'q3'):
            row[f'{m}_{q}'] = s['metrics'][m][q]

    compact.append(row)

write_csv('summary.csv', compact)
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')

for method, marker, color, offset in [
    ('milp', 'o', '#111111', -0.06),
    ('ga', 's', '#777777', 0.06)
]:
    group = [s for s in summary if s['method'] == method]
    for ax, key, label in [
        (axes[0], 'image_decode_ready_ms', 'Time until all images ready (s)'),
        (axes[1], 'image_requests_complete_ms', 'Time until final image byte (s)')
    ]:
        med = [s['metrics'][key]['median'] / 1000 for s in group]
        lo = [(s['metrics'][key]['median'] - s['metrics'][key]['q1']) / 1000 for s in group]
        hi = [(s['metrics'][key]['q3'] - s['metrics'][key]['median']) / 1000 for s in group]
        ax.errorbar(
            [s['scenario_mbps'] + offset for s in group],
            med,
            yerr=[lo, hi],
            marker=marker,
            color=color,
            capsize=5,
            label=method.upper(),
            linestyle='--' if method == 'ga' else '-'
        )
        ax.set(xlabel='Emulated network (Mbps)', ylabel=label, xticks=[4, 8, 12])
        ax.grid(axis='y', alpha=0.2)

for ax in axes:
    ax.axhline(2.5, color='#999999', linestyle=':', label='2.5 s modeled allowance')
    ax.legend(fontsize=9)

fig.suptitle('Measured cold-load medians and quartiles · navigation origin')
fig.savefig(OUT / 'browser-timings.svg')
fig.savefig(OUT / 'browser-timings.png', dpi=160)
plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')

for s in summary:
    st = s['metrics']['image_decode_ready_ms']
    x = st['median'] / 1000
    y = s['average_ssim']['mean']
    ax.errorbar(
        [x],
        [y],
        xerr=[[(st['median'] - st['q1']) / 1000], [(st['q3'] - st['median']) / 1000]],
        fmt='o' if s['method'] == 'milp' else 'x',
        color='black' if s['method'] == 'milp' else '#777777',
        capsize=4
    )
    ax.annotate(
        f"{s['scenario_mbps']} Mbps {s['method'].upper()}",
        (x, y),
        xytext=(-10, 12 if s['method'] == 'milp' else -19),
        ha='right' if s['scenario_mbps'] != 12 else 'left',
        textcoords='offset points',
        fontsize=10
    )

ax.margins(x=0.18, y=0.18)
ax.set(
    xlabel='Time until all images ready, median [Q1,Q3] (s)',
    ylabel='Mean SSIM across the 30 recorded selections',
    title='Quality versus readiness · one selected batch'
)
ax.grid(alpha=0.2)
fig.savefig(OUT / 'quality-readiness.svg')
fig.savefig(OUT / 'quality-readiness.png', dpi=160)

plt.close(fig)


def sec(st):
    return '—' if st['median'] is None else f"{st['median'] / 1000:.3f} [{st['q1'] / 1000:.3f}, {st['q3'] / 1000:.3f}]"


def table(headers, rows):
    return '<div class="table"><table><thead><tr>' + ''.join((
        f'<th>{html.escape(str(v))}</th>'
        for v in headers
    )) + '</tr></thead><tbody>' + ''.join((
        '<tr>' + ''.join((f'<td>{html.escape(str(v))}</td>' for v in row)) + '</tr>'
        for row in rows
    )) + '</tbody></table></div>'


quality = []
timing = []
targets = []

for s in summary:
    b = s['selected_bytes']
    q = s['average_ssim']
    quality.append([
        s['scenario_mbps'],
        s['method'].upper(),
        f"{s['scenario_mbps'] * 1000000 * 25 // 80:,}",
        f"{b['minimum']:,}" if b['minimum'] == b['maximum'] else f"{b['minimum']:,}–{b['maximum']:,}",
        f"{q['mean']:.12f}",
        f"{q['minimum']:.12f}–{q['maximum']:.12f}"
    ])
    timing.append([
        s['scenario_mbps'],
        s['method'].upper(),
        sec(s['metrics']['image_requests_complete_ms']),
        sec(s['metrics']['image_decode_ready_ms']),
        sec(s['metrics']['fcp_ms']),
        sec(s['metrics']['lcp_ms']),
        f"{s['failures']}/30"
    ])
    targets.append([
        s['scenario_mbps'],
        s['method'].upper(),
        sec(s['estimated_payload_transfer_ms']),
        f"{s['navigation_ready_within_2500_ms']}/30",
        f"{s['batch_ready_within_2500_ms']}/30"
    ])

qt = table(
    [
        'Mbps',
        'Method',
        'Byte cap',
        'Selected bytes (range)',
        'Mean SSIM across trials',
        'SSIM range'
    ],
    quality
)
tt = table(
    [
        'Mbps',
        'Method',
        'Final image byte, s',
        'All images ready, s',
        'FCP, s',
        'LCP, s',
        'Failed / attempts'
    ],
    timing
)
targets = table(
    [
        'Mbps',
        'Method',
        'Estimated payload s',
        'Navigation readiness ≤2.5s',
        'Batch readiness ≤2.5s'
    ],
    targets
)
pt = table(
    [
        'Mbps',
        'Same selections /30',
        'Quality matches /30',
        'GA − MILP readiness, ms: median [Q1,Q3]'
    ],
    [[
        p['scenario_mbps'],
        p['same_selections'],
        p['quality_matches'],
        f"{p['readiness_ga_minus_milp_ms']['median']:.2f} [{p['readiness_ga_minus_milp_ms']['q1']:.2f}, {p['readiness_ga_minus_milp_ms']['q3']:.2f}]"
    ] for p in paired_summary]
)
lt = table(
    ['Mbps', 'Method', 'Observed LCP elements (count)'],
    [[s['scenario_mbps'], s['method'].upper(), json.dumps(s['lcp_elements'])] for s in summary]
)

e = meta['environment']
failed = [
    {'run_id': r['run_id'], 'reasons': r['failure_reasons']}
    for r in runs
    if not r['success']
]
body = f"""<h1>Part 1: measured browser loading</h1>\n<p class="lead">A real local Node.js server served the unchanged image files. We compared recorded MILP and original GA selections in 180 cold-cache browser loads.</p>\n<p>This is a new visible-gallery lab experiment. All previous experiments remain unchanged.</p>\n<h2>What was compared</h2><p>At each speed, 30 recorded MILP trials were paired with the original 30 GA trials. Each saved selection was loaded once. This tests the original GA output distribution, not 30 repeats of one chosen GA selection. All selections meet the unchanged 0.90 featured / 0.70 ordinary floors and their original 2.5-second byte budgets. No optimizer ran during loading.</p>{qt}\n<p>At 4 Mbps, GA matched MILP in 23/30 trials. The seven other GA selections use 1,248,818 bytes, versus MILP’s 1,248,072 bytes: 746 extra bytes and 0.0000227078706465 less mean SSIM. At 8 and 12 Mbps both methods use identical image variants in every pair. Their timing differences cannot establish an image-selection speed advantage.</p>\n<h2>Measured times</h2><p>Values are median [first quartile, third quartile] in seconds, from navigation start. Successful loads only; every failed attempt is counted.</p>{tt}\n<img src="browser-timings.svg" alt="Medians and quartiles for image readiness and final image byte">\n<p>Positive paired differences mean GA was slower; negative means GA was faster. Small differences between identical selections are browser and host variability. This is a descriptive comparison, not a significance or equivalence test.</p>{pt}\n<img src="quality-readiness.svg" alt="SSIM versus measured readiness with quartile intervals">\n<h2>Does the 2.5-second allowance predict readiness?</h2><p>The allowance determines the byte cap. It is not a browser timing constraint. Estimated payload time excludes headers, document/CSS requests, latency, scheduling and decode. Navigation readiness starts at navigation; batch readiness starts at the first image fetch.</p>{targets}\n<h2>What produced LCP?</h2>{lt}<p>The page heading, “Fourteen images, one shared connection,” produced LCP in all 180 loads. LCP is the last observed largest-contentful-paint entry before the cutoff, not all-image readiness. All 14 images were in the viewport; the actual LCP element is recorded rather than assumed.</p>\n<h2>Controlled method</h2><ul>\n<li>4, 8 or 12 decimal Mbps download and upload; 100 ms emulated request-to-response-header latency. Shared Chrome network emulation, not separate bandwidth per image.</li>\n<li>Viewport 1440×1100, device scale 1, CPU throttle 1 (no slowdown), same headless browser and Node HTTP/1.1 server processes.</li>\n<li>Fresh context each load; HTTP cache disabled/cleared, service workers blocked/bypassed, server no-store. OS/filesystem caches were not cleared; startup file verification warms them.</li>\n<li>Same seven-column/two-row layout, native eager fetching, async decode, automatic priority, ascending IDs. No unused variants or preloading. Selected URL lengths can change document bytes slightly; the shell and CSS are fixed.</li>\n<li>Rotate network order per repetition. MILP first on even repetitions, GA first on odd. No retries or favorable-run selection.</li>\n<li>Verify exactly 14 selected requests in browser and server logs, HTTP200, file bytes, dimensions, decode, visible stable layout, paint metrics and absence of cache/errors.</li>\n<li>Observe 1000 ms after all decode promises settle. Timeout 30 s. Three instrumentation checks are saved separately in pilot/ and excluded.</li></ul>\n<h2>Metric definitions</h2><ul>\n<li>Final image byte: maximum image responseEnd from navigation.</li><li>All images ready: latest decode() resolution, called after each load event. This is not a guarantee every pixel has painted.</li><li>Batch span: final response or readiness minus earliest image fetch start, including queueing/latency.</li><li>FCP: first-contentful-paint. LCP: last observed largest-contentful-paint; tag/ID/text/URL retained.</li><li>Payload bytes: encodedBodySize checked against files. transferSize is saved separately and is not physical-link traffic.</li><li>Solver runtime is copied from the original experiment, dated separately; neither solver is rerun or included in browser time.</li></ul>\n<h2>Conclusions and limits</h2><p>All 60 loads at 12 Mbps reached image readiness within 2.5 seconds; none of the 120 loads at 4 or 8 Mbps did. The 4 and 8 Mbps selections use almost all of their respective payload allowance, while the all-high selection at 12 Mbps leaves 800,424 bytes unused. This explains why a faster network does not proportionally reduce readiness when the optimizer spends its larger budget on quality.</p><p>MILP has a small quality advantage in seven 4 Mbps pairs. At 8 and 12 Mbps, quality and selected files tie. These results do not demonstrate a universal loading advantage. One batch, one machine and localhost with network emulation cannot establish real-world performance. Smaller payloads alone do not prove better Core Web Vitals.</p>\n<p>Failed attempts: {len(failed)} of 180. Input and measurement-code hashes stayed unchanged. Every attempt is retained.</p>\n<details><summary>Environment, evidence and reproduction</summary>\n<p>Started {meta['started_at_utc']}; finished {meta['finished_at_utc']}. {html.escape(e['browser']['product'])}; Playwright {e['playwright']}; Node {e['node']}. {html.escape(e['cpu_model'])}, {e['logical_cpus']} logical CPUs, {e['total_memory_bytes'] / 1024 ** 3:.2f} GiB RAM; {html.escape(e['os_version'])} ({html.escape(e['os_release'])}). GPU details, host load and exact CDP commands are saved.</p>\n<p><a href="raw.csv">Raw CSV</a> · <a href="raw.json">Full raw JSON</a> · <a href="summary.json">Summary JSON</a> · <a href="paired.csv">Paired CSV</a> · <a href="plan.json">Executed plan and hashes</a> · <a href="metadata.json">Environment and definitions</a></p>\n<p>Sources: revisions/user-selected-v1/manifest.json and config.json; revisions/part1-milp-ga-v1/results/raw.json. Original match tolerance 10⁻¹⁰. No re-encoding or SSIM recomputation.</p>\n<pre>cd /home/kas/op\nnode revisions/part1-browser-benchmark-v1/server.mjs\n# New output directory required for another campaign:\nnode revisions/part1-browser-benchmark-v1/run.mjs --output revisions/part1-browser-benchmark-v1/reproduction-01\n.cache/report-venv/bin/python revisions/part1-browser-benchmark-v1/report.py revisions/part1-browser-benchmark-v1/reproduction-01</pre>\n<p>Reference: <a href="https://chromedevtools.github.io/devtools-protocol/tot/Network/">Chrome DevTools Network protocol</a>. Supported deprecated emulateNetworkConditions is explicitly recorded, with no silent fallback.</p></details>"""
style = 'body{font:18px/1.6 system-ui,sans-serif;color:#171717;max-width:1200px;margin:40px auto;padding:0 24px}h1{font-size:38px}h2{margin-top:40px}.lead{font-size:23px}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:16px}th,td{text-align:left;padding:10px;border-bottom:1px solid #bbb;vertical-align:top}th{border-bottom:2px solid black}img{width:100%;height:auto;margin:24px 0}details{border-top:1px solid;padding:20px 0}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}a{color:#184b6d}@media print{body{font-size:11pt;margin:0;max-width:none}h1{font-size:24pt}h2{break-after:avoid}table,img{break-inside:avoid}.table{overflow:visible}th,td{font-size:9pt;padding:6px}}'
(OUT / 'report.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Part1 browser benchmark</title><style>{style}</style><body>{body}</body></html>')

(OUT / 'failures.json').write_text(json.dumps(failed, indent=2) + '\n')

for s in summary:
    print(s['scenario_mbps'], s['method'], s['metrics']['image_decode_ready_ms'], s['lcp_elements'])

print('Paired:', json.dumps(paired_summary))
