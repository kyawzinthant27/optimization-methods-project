"""Post-hoc OFAT sensitivity using the original, unmodified Python GA."""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import os
import statistics as stats
import sys
import time
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
from image_quality.common import read_json, sha256, thresholds, validate_candidates
from image_quality.optimize import feasibility
from image_quality.run import environment
from course.run_ab import validate_record


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_save(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_ga(plan):
    spec = importlib.util.spec_from_file_location(
        'frozen_part1_ga',
        ROOT / plan['original_experiment'] / 'ga.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.solve_ga


def settings_for(plan, item):
    return {
        **plan['fixed_ga'],
        'mutation_probability': item['mutation_probability'],
        'elitism': item['elitism']
    }


def preservation_snapshot(plan):
    paths = {
        ROOT / p
        for p in read_json(ROOT / plan['original_experiment'] / 'preserved-before.json')
    }

    for folder in [ROOT / 'revisions', ROOT / 'reports', ROOT / 'image_quality']:
        paths.update((
            p
            for p in folder.rglob('*')
            if p.is_file() and BASE not in p.parents and ('__pycache__' not in p.parts)
        ))

    return {str(p): sha256(p) for p in sorted(paths)}


def verify_hashes(hashes):
    changed = [p for p, h in hashes.items() if not Path(p).is_file() or sha256(p) != h]
    if changed:
        raise ValueError('Preserved files changed: ' + ', '.join(changed))


def summarize(records, plan, phase):
    summaries = []

    for config in plan['configurations']:
        group = [r for r in records if r['phase'] == phase and r['configuration'] == config['id']]
        if not group:
            continue

        valid = [r for r in group if r['success'] and r['feasible']]
        gaps = [r['absolute_ssim_gap'] for r in valid]
        times = [r['solve_time_ms'] for r in valid]
        summaries.append({
            'phase': phase,
            'configuration': config['id'],
            'mutation_probability': config['mutation_probability'],
            'elitism': config['elitism'],
            'runs': len(group),
            'feasible_runs': sum((r['feasible'] for r in group)),
            'infeasible_runs': sum((not r['feasible'] for r in group)),
            'failed_runs': sum((not r['success'] for r in group)),
            'matches': sum((r['matches_milp'] for r in group)),
            'match_rate': sum((r['matches_milp'] for r in group)) / len(group),
            'mean_ssim_gap': stats.mean(gaps) if gaps else None,
            'median_ssim_gap': stats.median(gaps) if gaps else None,
            'worst_ssim_gap': max(gaps) if gaps else None,
            'median_runtime_ms': stats.median(times) if times else None,
            'min_runtime_ms': min(times) if times else None,
            'max_runtime_ms': max(times) if times else None,
            'rank': None
        })

    return summaries


def rank(summaries, plan):
    order = {c['id']: i for i, c in enumerate(plan['configurations'])}
    ordered = sorted(
        summaries,
        key=lambda s: (
            -s['matches'],
            float('inf') if s['failed_runs'] or s['infeasible_runs'] else s['mean_ssim_gap'],
            s['median_runtime_ms'] if s['median_runtime_ms'] is not None else float('inf'),
            order[s['configuration']]
        )
    )

    for n, summary in enumerate(ordered, 1):
        summary['rank'] = n

    return ordered


def export_raw(out, records, metadata):
    save(out / 'raw.json', {'metadata': metadata, 'runs': records})
    flat = []

    for r in records:
        flat.append({
            **{k: r.get(k) for k in [
                'execution_index',
                'phase',
                'configuration',
                'seed',
                'scenario_mbps',
                'budget_bytes',
                'started_at_utc',
                'success',
                'feasible',
                'status',
                'total_bytes',
                'unused_budget_bytes',
                'average_ssim',
                'matches_milp',
                'absolute_ssim_gap',
                'solve_time_ms',
                'elapsed_ns',
                'generations_completed',
                'fitness_evaluations',
                'repair_calls',
                'repair_moves'
            ]},
            **r['settings'],
            'selected_versions': json.dumps({
                c['image_id']: c['version']
                for c in r.get('selections', [])
            }),
            'error': r.get('error', r.get('validation_error', ''))
        })

    if flat:
        csv_save(out / 'raw.csv', flat)


def run(out):
    plan = read_json(BASE / 'plan.json')
    if out.exists():
        raise FileExistsError('Refusing existing output directory: ' + str(out))

    if BASE not in out.parents:
        raise ValueError('Use a new output directory inside this revision')

    for key, value in plan['thread_environment'].items():
        if os.environ.get(key) != value:
            raise ValueError('Set ' + key + '=' + value + ' before starting Python')

    for p, h in plan['input_hashes'].items():
        if sha256(ROOT / p) != h:
            raise ValueError('Input changed since saved plan: ' + p)

    dataset = ROOT / plan['dataset']

    for p, h in read_json(dataset / 'freeze.json')['files_sha256'].items():
        if sha256(ROOT / p) != h:
            raise ValueError('Frozen dataset changed: ' + p)

    rows, cfg = (read_json(dataset / 'manifest.json'), read_json(dataset / 'config.json'))
    limits = thresholds(cfg)
    validate_candidates(rows, cfg['image_ids'])
    assert len(cfg['image_ids']) == 14 and len(rows) == 42
    assert set(cfg['important_thresholds']) == set(plan['quality_floors']['important_ids'])
    assert all((t == (0.9 if i in cfg['important_thresholds'] else 0.7) for i, t in limits.items()))

    for row in rows:
        assert (ROOT / row['path']).stat().st_size == row['bytes']
        assert sha256(ROOT / row['path']) == row['sha256']

    old = read_json(ROOT / plan['original_experiment'] / 'results/raw.json')
    original_ga = {
        r['seed']: r
        for r in old['runs']
        if r['method'] == 'ga' and r['scenario_mbps'] == 4
    }
    assert sorted(original_ga) == plan['tuning_seeds']
    assert not set(plan['fresh_seeds']) & {r['seed'] for r in old['runs'] if r['seed'] is not None}
    assert not set(plan['fresh_seeds']) & set(plan['tuning_seeds'])
    milps = [r for r in old['runs'] if r['method'] == 'milp' and r['scenario_mbps'] == 4]

    for m in milps:
        validate_record(m, rows, limits, plan['budget_bytes'])
        assert m['status'] == 'optimal' and m['optimality_gap'] == 0
        assert m['average_ssim'] == plan['milp_reference_ssim']

    preflight = feasibility(rows, limits)
    assert preflight['minimum_feasible_bytes'] == 891592
    ga = load_ga(plan)
    preserved = preservation_snapshot(plan)
    out.mkdir(parents=True)
    (out / 'trials').mkdir()
    save(out / 'preserved-before.json', preserved)
    current_environment = environment()
    differences = {
        k: {'original': old['metadata']['environment'].get(k), 'current': v} for k,
        v in current_environment.items() if v != old['metadata']['environment'].get(k)
    }
    metadata = {
        'started_at_utc': now(),
        'completed': False,
        'plan_sha256': sha256(BASE / 'plan.json'),
        'plan_saved_at_utc': plan['saved_at_utc'],
        'environment': current_environment,
        'original_environment_differences': differences,
        'timing': plan['timing'],
        'input_hashes': plan['input_hashes'],
        'runner_sha256': sha256(Path(__file__)),
        'milp_reference_ssim': plan['milp_reference_ssim'],
        'match_tolerance': plan['match_tolerance'],
        'preflight': preflight,
        'preserved_files': len(preserved)
    }
    assert metadata['started_at_utc'] > plan['saved_at_utc']
    save(out / 'execution.json', metadata)
    records, warmups = ([], [])

    def execute(config, seed, phase, warmup=False):
        settings = settings_for(plan, config)
        started = now()
        begin = time.perf_counter_ns()

        try:
            result = ga(rows, limits, plan['budget_bytes'], settings, seed)
            elapsed = time.perf_counter_ns() - begin
        except Exception as exc:
            elapsed = time.perf_counter_ns() - begin
            result = {
                'success': False,
                'feasible': False,
                'status': 'exception',
                'error': repr(exc),
                'selections': []
            }

        r = {
            **result,
            'phase': phase,
            'configuration': config['id'],
            'settings': settings,
            'seed': seed,
            'scenario_mbps': 4,
            'budget_bytes': plan['budget_bytes'],
            'started_at_utc': started,
            'elapsed_ns': elapsed,
            'solve_time_ms': elapsed / 1000000.0,
            'success': False,
            'matches_milp': False,
            'absolute_ssim_gap': None
        }
        if r['status'] != 'exception':
            try:
                validate_record(r, rows, limits, plan['budget_bytes'])
                assert r['average_ssim'] <= plan['milp_reference_ssim'] + plan['match_tolerance']
                assert r['generations_completed'] == 200 and r['fitness_evaluations'] == 20100
                assert len(r['history']) == 201
                r['success'] = bool(r['feasible'] and r['status'] == 'completed_heuristic')
            except Exception as exc:
                r['validation_error'] = repr(exc)

        r['unused_budget_bytes'] = plan['budget_bytes'] - r['total_bytes'] if r.get('total_bytes') is not None else None
        if r['success']:
            r['absolute_ssim_gap'] = abs(plan['milp_reference_ssim'] - r['average_ssim'])
            r['matches_milp'] = r['absolute_ssim_gap'] <= plan['match_tolerance']

        if warmup:
            warmups.append(r)
            save(out / 'warmups.json', warmups)
        else:
            r['execution_index'] = len(records)
            records.append(r)
            save(out / 'trials' / f"{phase}-{config['id']}-{seed}.json", r)

        return r

    configs = plan['configurations']

    for config in configs:
        if not execute(config, 900000, 'warmup_tuning', True)['success']:
            raise ValueError('Warmup failed; outputs retained; inspect before proceeding')

    for seed in plan['tuning_seeds']:
        execute(configs[0], seed, 'tuning')

    fields = [
        'feasible',
        'status',
        'total_bytes',
        'average_ssim',
        'genes',
        'selections',
        'history',
        'generations_completed',
        'fitness_evaluations',
        'repair_calls',
        'repair_moves'
    ]
    mismatches = [{
        'seed': r['seed'],
        'different_fields': [k for k in fields if r.get(k) != original_ga[r['seed']].get(k)]
    } for r in records if any((r.get(k) != original_ga[r['seed']].get(k) for k in fields))]
    baseline_matches = sum((r['matches_milp'] for r in records))
    baseline = {
        'checked_at_utc': now(),
        'matches': baseline_matches,
        'expected_matches': 23,
        'deterministic_fields_compared': fields,
        'mismatches': mismatches,
        'environment_differences': differences,
        'passed': baseline_matches == 23 and (not mismatches) and all((
            r['success']
            for r in records
        )),
        'note': 'Computation times are newly measured and are not expected to reproduce original times.'
    }
    save(out / 'baseline-reproduction.json', baseline)
    export_raw(out, records, metadata)

    if not baseline['passed']:
        raise ValueError('Baseline did not reproduce; diagnosis saved; sweep and fresh evaluation not run')

    print(
        'Baseline reproduced: 23/30 matches; all 30 final outputs, histories and counters identical.',
        flush=True
    )
    alternatives = configs[1:]

    for n, seed in enumerate(plan['tuning_seeds']):
        order = alternatives[n % 4:] + alternatives[:n % 4]
        for config in order:
            execute(config, seed, 'tuning')

        if (n + 1) % 5 == 0:
            print(
                f'Sensitivity: {n + 1}/30 paired seeds ({len(records)} measured calls)',
                flush=True
            )

    ranked = rank(summarize(records, plan, 'tuning'), plan)
    candidates = [
        s
        for s in ranked
        if s['configuration'] != 'baseline' and (not s['failed_runs']) and (not s['infeasible_runs'])
    ]

    if not candidates:
        export_raw(out, records, metadata)
        raise ValueError('No valid non-baseline configuration for fresh evaluation')

    winner = next((c for c in configs if c['id'] == candidates[0]['configuration']))
    selection = {
        'selected_at_utc': now(),
        'criterion': plan['ranking'],
        'ranking': ranked,
        'top_nonbaseline': winner,
        'fresh_seeds': plan['fresh_seeds'],
        'fresh_results_used_in_selection': False
    }
    save(out / 'selection-before-fresh.json', selection)
    export_raw(out, records, metadata)
    print('Top non-baseline selected before fresh evaluation: ' + winner['id'], flush=True)
    pair = [configs[0], winner]

    for config in pair:
        if not execute(config, 900001, 'warmup_fresh', True)['success']:
            raise ValueError('Fresh-stage warmup failed; retained, not retried')

    for n, seed in enumerate(plan['fresh_seeds']):
        for config in pair if n % 2 == 0 else pair[::-1]:
            execute(config, seed, 'fresh')

        if (n + 1) % 10 == 0:
            print(f'Fresh evaluation: {n + 1}/30 paired seeds', flush=True)

    summaries = ranked + summarize(records, plan, 'fresh')
    save(out / 'summary.json', summaries)
    csv_save(out / 'summary.csv', summaries)
    verify_hashes(preserved)
    assert sha256(BASE / 'plan.json') == metadata['plan_sha256']
    assert sha256(Path(__file__)) == metadata['runner_sha256']
    assert environment() == current_environment
    metadata.update(
        completed=True,
        completed_at_utc=now(),
        measured_calls=len(records),
        warmup_calls=len(warmups),
        preserved_files_unchanged=len(preserved)
    )
    export_raw(out, records, metadata)
    save(out / 'execution.json', metadata)
    from report import build_report

    build_report(out, plan)
    print('Completed 150 tuning + 60 fresh calls. Reports: ' + str(out), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE / 'results')
    args = parser.parse_args()
    run(args.output.resolve())
