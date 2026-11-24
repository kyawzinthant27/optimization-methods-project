from datetime import datetime, timezone
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from statistics import median
import csv
import json
import os
import sys
import time

base = Path(__file__).resolve().parent
root = base.parents[1]
sys.path.insert(0, str(root))
from image_quality.common import thresholds, validate_candidates
from course.run_ab import validate_record
from image_quality.run import environment


def read(p):
    return json.loads((root / p).read_text())


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def digest(p):
    return sha256((root / p).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


original = Path('revisions/part1-milp-ga-v1')
sensitivity = Path('revisions/part1-ga-sensitivity-v1')
dataset = Path('revisions/user-selected-v1')
prior = read(sensitivity / 'plan.json')
old = read(original / 'results/raw.json')

prior_exp = read(sensitivity / 'results/raw.json')
manifest = read(dataset / 'manifest.json')
config = read(dataset / 'config.json')
limits = thresholds(config)
validate_candidates(manifest, config['image_ids'])

assert len(manifest) == 42 and len(limits) == 14

for k, v in prior['thread_environment'].items():
    assert os.environ.get(k) == v

for p, h in prior['input_hashes'].items():
    assert digest(p) == h, p

for row in manifest:
    assert digest(row['path']) == row['sha256']
    assert (root / row['path']).stat().st_size == row['bytes']

settings = {**prior['fixed_ga'], 'mutation_probability': 0.1, 'elitism': 2}
budget = prior['budget_bytes']
target = prior['milp_reference_ssim']
tol = prior['match_tolerance']
refs = [r for r in old['runs'] if r['method'] == 'milp' and r['scenario_mbps'] == 4]

assert len(refs) == 30
assert all((
    r['status'] == 'optimal' and r['optimality_gap'] == 0 and (r['budget_bytes'] == budget) and (r['average_ssim'] == target)
    for r in refs
))
assert any((r['settings'] == settings and r['matches_milp'] for r in prior_exp['runs']))
seeds = list(range(30000, 30030))
warmup_seed = 930000
assert not set(seeds + [warmup_seed]) & {r.get('seed') for r in old['runs'] + prior_exp['runs']}
schedule = [(s, g) for i, s in enumerate(seeds) for g in ((100, 200) if i % 2 == 0 else (200, 100))]
paths = list(prior['input_hashes']) + [prior_i for prior_i in map(
    str,
    [sensitivity / 'plan.json', sensitivity / 'results/raw.json', Path(__file__).relative_to(root)]
)] + [r['path'] for r in manifest]
hashes = {str(p): digest(p) for p in paths}
out = base / 'results'
out.mkdir()

plan = {
    'saved_at_utc': now(),
    'seeds': seeds,
    'warmup_seed': warmup_seed,
    'budget_bytes': budget,
    'scenario_mbps': 4,
    'target_mean_ssim': target,
    'absolute_match_tolerance': tol,
    'fixed_ga': {k: v for k, v in settings.items() if k != 'generations'},
    'schedule': schedule,
    'order': 'Alternates 100-first and 200-first, runs paired by seed.',
    'timing': 'Call-to-return of untouched solve_ga. Excludes validation and I/O.',
    'source_hashes': hashes,
    'environment': environment()
}
save(out / 'plan.json', plan)
spec = spec_from_file_location('original_ga', root / original / 'ga.py')
ga = module_from_spec(spec)
spec.loader.exec_module(ga)


def run(seed, gen):
    cfg = {**settings, 'generations': gen}
    start = time.perf_counter_ns()
    raw = ga.solve_ga(manifest, limits, budget, cfg, seed)
    elapsed = time.perf_counter_ns() - start
    validation_error = None

    try:
        validate_record(raw, manifest, limits, budget)
        assert raw['feasible'] and raw['status'] == 'completed_heuristic'
        assert raw['generations_completed'] == gen and len(raw['history']) == gen + 1
        assert raw['fitness_evaluations'] == (gen + 1) * settings['population']
        assert raw['average_ssim'] <= target + tol
    except Exception as e:
        validation_error = repr(e)

    gap = abs(raw['average_ssim'] - target) if validation_error is None else None
    first = next(
        (h['generation'] for h in raw['history'] if abs(h['best_ssim'] - target) <= tol),
        None
    )

    return {
        **raw,
        'settings': cfg,
        'elapsed_ns': elapsed,
        'runtime_ms': elapsed / 1000000.0,
        'validation_error': validation_error,
        'absolute_gap': gap,
        'matches_milp': gap is not None and gap <= tol,
        'first_match_generation': first
    }


warmups = [run(warmup_seed, g) for g in (100, 200)]
save(out / 'warmups.json', warmups)
assert all((r['validation_error'] is None for r in warmups))

with (out / 'raw.jsonl').open('x') as log:
    records = []
    for seed, g in schedule:
        r = run(seed, g)
        records.append(r)
        log.write(json.dumps(r, allow_nan=False) + '\n')
        log.flush()

assert all((digest(p) == h for p, h in hashes.items()))
by = {(r['seed'], r['generations_completed']): r for r in records}
assert len(by) == 60
assert all((r['validation_error'] is None for r in records))
assert all((by[s, 100]['history'] == by[s, 200]['history'][:101] for s in seeds))
assert all((not by[s, 100]['matches_milp'] or by[s, 200]['matches_milp'] for s in seeds))
summary = []

for g in (100, 200):
    group = [by[s, g] for s in seeds]
    misses = [r['seed'] for r in group if not r['matches_milp']]
    summary.append({
        'generations': g,
        'trials': len(group),
        'feasible': sum((r['feasible'] for r in group)),
        'matches': sum((r['matches_milp'] for r in group)),
        'miss_seeds': misses,
        'median_runtime_ms': median((r['runtime_ms'] for r in group)),
        'worst_mean_ssim_gap': max((r['absolute_gap'] for r in group)),
        'latest_first_match_generation': max(
            (r['first_match_generation'] for r in group if r['first_match_generation'] is not None),
            default=None
        ),
        'fitness_evaluations_per_trial': group[0]['fitness_evaluations']
    })

a, b = summary
save(
    out / 'summary.json',
    {
        'results': summary,
        'all_source_hashes_unchanged': True,
        'all_30_paired_history_prefixes_match': True,
        'median_runtime_reduction_pct': 100 * (1 - a['median_runtime_ms'] / b['median_runtime_ms']),
        'median_paired_runtime_saving_ms': median((
            by[s, 200]['runtime_ms'] - by[s, 100]['runtime_ms']
            for s in seeds
        )),
        'seeds_rescued_after_generation_100': [
            s
            for s in seeds
            if not by[s, 100]['matches_milp'] and by[s, 200]['matches_milp']
        ]
    }
)

with (out / 'trials.csv').open('x') as f:
    keys = [
        'seed',
        'generations_completed',
        'feasible',
        'total_bytes',
        'average_ssim',
        'matches_milp',
        'absolute_gap',
        'first_match_generation',
        'fitness_evaluations',
        'runtime_ms'
    ]
    writer = csv.DictWriter(f, keys, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(records)

print((out / 'summary.json').read_text())
