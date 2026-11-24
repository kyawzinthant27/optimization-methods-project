from pathlib import Path
from statistics import median, quantiles
from collections import defaultdict
import csv
import json

base = Path(__file__).resolve().parent
out = base / 'results'
cases = json.loads((base / 'cases.json').read_text())
data = json.loads((out / 'raw.json').read_text())
plan = json.loads((out / 'plan.json').read_text())

runs = data['runs']
assert data['metadata']['inputs_unchanged']
assert len(runs) == len(plan['plan'])

for observed, expected in zip(runs, plan['plan'], strict=True):
    assert all((observed[k] == v for k, v in expected.items()))

groups = defaultdict(list)
refs = {}

for r in runs:
    groups[r['method']].append(r)
    if r['success'] and r['method'] == cases['caps'][-1]['browser_case']:
        refs[r['round']] = r['metrics']['image_decode_ready_ms']

original = json.loads((base.parent / 'part1-browser-benchmark-v1/results/summary.json').read_text())
original_median = next((
    g
    for g in original['groups']
    if g['method'] == 'milp' and g['scenario_mbps'] == 4
))['metrics']['image_decode_ready_ms']['median']


def stats(values):
    if not values:
        return None

    q1, _, q3 = quantiles(values, n=4, method='inclusive') if len(values) > 1 else (values[0],) * 3
    return {'median_ms': median(values), 'q1_ms': q1, 'q3_ms': q3, 'fastest_ms': min(values)}


summary = []

for row in cases['caps']:
    group = groups.get(row.get('browser_case'), [])
    ok = [r for r in group if r['success']]

    for r in ok:
        assert r['selected_bytes'] == r['metrics']['image_payload_bytes'] == row['total_bytes']
        assert r['average_ssim'] == row['average_ssim']

    diffs = [
        r['metrics']['image_decode_ready_ms'] - refs[r['round']]
        for r in ok
        if r['round'] in refs
    ]
    ready = stats([r['metrics']['image_decode_ready_ms'] for r in ok])
    response = stats([r['metrics']['image_requests_complete_ms'] for r in ok])
    summary.append({
        **row,
        'attempts': len(group),
        'successes': len(ok),
        'readiness': ready,
        'last_image_response': response,
        'paired_vs_reference_median_ms': median(diffs) if diffs else None,
        'paired_valid_rounds': len(diffs)
    })

failures = [{
    'run_id': r['run_id'],
    'case': r['method'],
    'reasons': r['failure_reasons']
} for r in runs if not r['success']]
summary_obj = {
    'all_attempts': len(runs),
    'failures': failures,
    'original_4mbps_milp_readiness_median_ms': original_median,
    'min_feasible_bytes': cases['min_feasible_bytes'],
    'min_bytes_at_reference_quality': cases['min_bytes_at_reference_quality'],
    'cap_results': summary
}
(out / 'summary.json').write_text(json.dumps(summary_obj, indent=2) + '\n')
fields = [
    'cap_bytes',
    'feasible_assignments',
    'total_bytes',
    'average_ssim',
    'payload_estimate_ms',
    'browser_case',
    'attempts',
    'successes',
    'median_ms',
    'q1_ms',
    'q3_ms',
    'fastest_ms',
    'last_response_median_ms',
    'paired_vs_reference_median_ms',
    'paired_valid_rounds'
]

with (out / 'summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()

    for r in summary:
        record = {
            **r,
            **(r['readiness'] or {}),
            'last_response_median_ms': (r['last_image_response'] or {}).get('median_ms')
        }
        writer.writerow({k: record.get(k) for k in fields})

eligible = [r for r in summary if r['readiness']]
floor, ref = (summary[1], summary[-1])
fastest = min(eligible, key=lambda r: r['readiness']['median_ms'])


def sec(ms):
    return f'{ms / 1000:.3f}'


lines = [
    '# 4 Mbps stress test',
    '',
    f'Chrome used the original 14-image gallery and its 4 Mbps bandwidth cap and 100 ms emulated latency. {len(runs) - len(failures)}/{len(runs)} loads passed validation. Timing starts at navigation and ends when all images decode.',
    '',
    '| Byte cap | Feasible assignments | Selected bytes | Mean SSIM | Payload estimate (s) | Median ready (s) | Q1–Q3 (s) | Fastest load (s) |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|'
]

for r in summary:
    if 'total_bytes' not in r:
        lines.append(f"| {r['cap_bytes']:,} | 0 | N/A | N/A | N/A | N/A | N/A | N/A |")
        continue

    a = r['readiness']
    timings = f"{sec(a['median_ms'])} | {sec(a['q1_ms'])}–{sec(a['q3_ms'])} | {sec(a['fastest_ms'])}" if a else 'N/A | N/A | N/A'
    lines.append(f"| {r['cap_bytes']:,} | {r['feasible_assignments']:,} | {r['total_bytes']:,} | {r['average_ssim']:.6f} | {sec(r['payload_estimate_ms'])} | {timings} |")

lines += [
    '',
    f"- {floor['cap_bytes']:,} bytes is the exact feasibility boundary. One byte less leaves no feasible selection. The 900,000-byte cap returns the same files, so it shares the measurements.",
    f"- The smallest payload that retains {ref['average_ssim']:.9f} mean SSIM is {cases['min_bytes_at_reference_quality']:,} bytes under a 1e-12 SSIM tolerance.",
    f"- The fastest median among tested selections was {sec(fastest['readiness']['median_ms'])} s at {fastest['total_bytes']:,} bytes and {fastest['average_ssim']:.6f} mean SSIM. Across all valid runs, the fastest individual load was {sec(min((
        r['readiness']['fastest_ms']
        for r in eligible
    )))} s.",
    f"- The same-campaign reference median was {sec(ref['readiness']['median_ms'])} s. The earlier benchmark reported {sec(original_median)} s; the campaigns are kept separate.",
    '- The exact byte bound is specific to these candidates and SSIM floors. Browser performance is measured under the controlled local benchmark, not a guarantee on a remote network.'
]
(out / 'report.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
