import csv
import json
from itertools import product
from math import prod
from pathlib import Path

script_directory = Path(__file__).resolve().parent
root = script_directory.parents[1]

input_directory = root / 'revisions/user-selected-v1'
manifest = json.loads((input_directory / 'manifest.json').read_text())
config = json.loads((input_directory / 'config.json').read_text())

# Filter out versions below the SSIM floor for each image.
eligible = []

for image_id in sorted(config['image_ids']):
    floor = config['important_thresholds'].get(image_id, config['ordinary_threshold'])
    options = [
        candidate
        for candidate in manifest
        if candidate['image_id'] == image_id and candidate['ssim'] >= floor
    ]
    assert options
    eligible.append([
        (option['bytes'], option['ssim'], option)
        for option in options
    ])

# The cheapest eligible version of each image sets the feasibility limit.
minimum = sum(min(option[0] for option in row) for row in eligible)
byte_caps = [minimum - 1, minimum, 900_000, 950_000, 1_000_000, 1_125_000, 1_200_000, 1_250_000]
best_by_cap = [None] * len(byte_caps)
feasible_counts = [0] * len(byte_caps)
best_by_bytes = {}

# Maximize mean SSIM, breaking ties in favor of fewer bytes.
for assignment in product(*eligible):
    size = sum(option[0] for option in assignment)
    if size > byte_caps[-1]:
        continue

    score = sum(option[1] for option in assignment) / len(eligible)
    row = (score, -size, assignment)
    best_by_bytes[size] = max(score, best_by_bytes.get(size, 0))

    for i, cap in enumerate(byte_caps):
        if size <= cap:
            feasible_counts[i] += 1
            if best_by_cap[i] is None or row[:2] > best_by_cap[i][:2]:
                best_by_cap[i] = row

assert best_by_cap[0] is None and feasible_counts[0] == 0
assert -best_by_cap[1][1] == minimum

# Check the shared budgets against the MILP results.
prior = root / 'revisions/part1-milp-budget-sensitivity-v1/results/summary.csv'

for old in csv.DictReader(prior.open()):
    i = byte_caps.index(int(old['byte_cap']))
    assert abs(best_by_cap[i][0] - float(old['selected_mean_ssim'])) <= 1e-12
    assert -best_by_cap[i][1] == int(old['bytes_used'])

minimum_bytes_at_reference_quality = min(
    size for size, score in best_by_bytes.items()
    if score >= best_by_cap[-1][0] - 1e-12
)
assert minimum_bytes_at_reference_quality == -best_by_cap[-1][1]

# Benchmark each distinct set of selected files only once.
cases, cap_results = [], []

for cap, entry, count in zip(byte_caps, best_by_cap, feasible_counts):
    row = {'cap_bytes': cap, 'feasible_assignments': count}
    cap_results.append(row)

    if entry is None:
        continue

    score, neg_bytes, selection = entry
    payload = -neg_bytes
    row.update(
        total_bytes=payload,
        average_ssim=score,
        payload_estimate_ms=payload / 500,
    )
    selected = [x[2] for x in selection]
    existing = next((c for c in cases if c['selections'] == selected), None)

    if existing:
        row['browser_case'] = existing['method']
        continue

    if cap == minimum:
        method = 'floor-min'
    elif cap == byte_caps[-1]:
        method = 'reference'
    else:
        method = f'cap-{cap}'

    row['browser_case'] = method
    cases.append({
        'method': method,
        'scenario_mbps': 4,
        'repetition': 0,
        'budget_bytes': cap,
        'total_bytes': payload,
        'average_ssim': score,
        'minimum_ssim': min(option[1] for option in selection),
        'selections': selected
    })

with (script_directory / 'cases.json').open('x') as f:
    json.dump(
        {
            'eligible_assignments': prod(map(len, eligible)),
            'min_feasible_bytes': minimum,
            'min_bytes_at_reference_quality': minimum_bytes_at_reference_quality,
            'quality_tolerance': 1e-12,
            'caps': cap_results,
            'browser_cases': cases
        },
        f,
        indent=2
    )

for row in cap_results:
    print(row)
