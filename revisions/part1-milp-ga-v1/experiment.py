"""Execute the saved Part 1 plan once. No data generation or prior-result changes."""
from pathlib import Path
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import milp
from image_quality.common import read_json, sha256, thresholds, validate_candidates, budget_bytes, write_json
from image_quality.optimize import solve, feasibility
from image_quality.run import environment
from course.run_ab import validate_record
from ga import solve_ga


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_preserved():
    hashes = read_json(BASE / 'preserved-before.json')

    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError('Existing artifact changed: ' + path)

    return len(hashes)


def run():
    plan = read_json(BASE / 'plan.json')
    out = BASE / 'results'

    if out.exists():
        raise FileExistsError('Results already exist; do not overwrite or repeat this campaign')

    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        if os.environ.get(key) != '1':
            raise ValueError('Set ' + key + '=1 before starting Python')

    preserved = verify_preserved()
    dataset = ROOT / plan['dataset']

    for path, expected in read_json(dataset / 'freeze.json')['files_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError('Frozen input changed: ' + path)

    rows = read_json(dataset / 'manifest.json')
    cfg = read_json(dataset / 'config.json')
    limits = thresholds(cfg)
    validate_candidates(rows, cfg['image_ids'])
    assert set(cfg['important_thresholds']) == set(plan['important_ids'])
    assert all((
        limits[i] == (
            plan['important_floor']
            if i in plan['important_ids'] else plan['ordinary_floor']
        )
        for i in limits
    ))
    assert [
        budget_bytes(r * 1000000, plan['payload_allowance_seconds'])
        for r in plan['network_mbps']
    ] == plan['budgets_bytes']

    for row in rows:
        assert sha256(ROOT / row['path']) == row['sha256']
        assert (ROOT / row['path']).stat().st_size == row['bytes']

    preflight = feasibility(rows, limits)
    assert preflight['minimum_feasible_bytes'] == 891592
    out.mkdir()
    (out / 'trials').mkdir()
    metadata = {
        'started_at_utc': now(),
        'completed': False,
        'environment': environment(),
        'plan_sha256': sha256(BASE / 'plan.json'),
        'plan_saved_at_utc': plan['saved_at_utc'],
        'manifest_sha256': sha256(dataset / 'manifest.json'),
        'preflight': preflight,
        'timing': plan['timing'],
        'preserved_files': preserved,
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [
            BASE / 'ga.py',
            BASE / 'experiment.py',
            ROOT / 'image_quality/optimize.py'
        ]}
    }
    assert metadata['started_at_utc'] > plan['saved_at_utc']
    write_json(out / 'execution.json', metadata)

    def execute(method, scenario, repetition, warmup=False):
        budget = plan['budgets_bytes'][scenario]
        seed = (
            900000 + scenario
            if warmup else 10000 + scenario * 1000 + repetition
        ) if method == 'ga' else None
        started = now()
        begin = time.perf_counter_ns()

        try:
            result = solve(rows, limits, budget, plan['solver']) if method == 'milp' else solve_ga(
                rows,
                limits,
                budget,
                plan['ga'],
                seed
            )
            elapsed = (time.perf_counter_ns() - begin) / 1000000000.0
        except Exception as exc:
            elapsed = (time.perf_counter_ns() - begin) / 1000000000.0
            return {
                'method': method,
                'scenario_mbps': plan['network_mbps'][scenario],
                'budget_bytes': budget,
                'repetition': repetition,
                'seed': seed,
                'warmup': warmup,
                'started_at_utc': started,
                'runtime_seconds': elapsed,
                'success': False,
                'feasible': False,
                'status': 'exception',
                'error': repr(exc)
            }

        record = dict(
            result,
            method=method,
            scenario_mbps=plan['network_mbps'][scenario],
            repetition=repetition,
            seed=seed,
            warmup=warmup,
            started_at_utc=started,
            runtime_seconds=elapsed,
            solver_only_seconds=result.get('solve_seconds')
        )

        try:
            validate_record(record, rows, limits, budget)
            record['success'] = record['feasible'] and record['status'] in (
                'optimal',
                'completed_heuristic'
            )
        except Exception as exc:
            record['success'] = False
            record['validation_error'] = repr(exc)

        record['unused_budget_bytes'] = budget - result['total_bytes'] if result['total_bytes'] is not None else None
        record['quality_margins'] = {
            r['image_id']: r['ssim'] - limits[r['image_id']]
            for r in result['selections']
        }

        return record

    warmups = [
        execute(method, scenario, -1, True)
        for scenario in range(3)
        for method in ['milp', 'ga']
    ]
    write_json(out / 'warmups.json', warmups)

    if any((not r['success'] for r in warmups)):
        raise ValueError('A warmup failed; inspect records before continuing')

    records = []

    for repetition in range(plan['repetitions_per_method_per_scenario']):
        for offset in range(3):
            scenario = (repetition + offset) % 3
            order = ['milp', 'ga'] if (repetition + scenario) % 2 == 0 else ['ga', 'milp']

            for method in order:
                record = execute(method, scenario, repetition)
                record['execution_index'] = len(records)
                records.append(record)
                write_json(
                    out / 'trials' / f"{plan['network_mbps'][scenario]}Mbps-{method}-{repetition:02d}.json",
                    record
                )

        print(
            f'Completed repetition {repetition + 1}/30 ({len(records)} measured calls)',
            flush=True
        )

    metadata.update(
        completed_at_utc=now(),
        completed=True,
        preserved_files_unchanged=verify_preserved()
    )
    assert metadata['plan_sha256'] == sha256(BASE / 'plan.json')

    for p, h in metadata['code_sha256'].items():
        assert sha256(ROOT / p) == h

    write_json(out / 'raw.json', {'metadata': metadata, 'runs': records})
    write_json(out / 'execution.json', metadata)
    fields = [
        'execution_index',
        'method',
        'scenario_mbps',
        'budget_bytes',
        'repetition',
        'seed',
        'success',
        'feasible',
        'status',
        'average_ssim',
        'minimum_ssim',
        'total_bytes',
        'unused_budget_bytes',
        'runtime_seconds',
        'solver_only_seconds',
        'optimality_gap',
        'solver_status_code',
        'generations_completed',
        'fitness_evaluations',
        'repair_moves'
    ]

    with (out / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    with (out / 'selections.csv').open('w', newline='') as stream:
        fields = [
            'method',
            'scenario_mbps',
            'repetition',
            'image_id',
            'version',
            'bytes',
            'ssim',
            'threshold',
            'quality_margin'
        ]
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()

        for r in records:
            for c in r.get('selections', []):
                writer.writerow({
                    **c,
                    **{k: r[k] for k in ['method', 'scenario_mbps', 'repetition']},
                    'threshold': limits[c['image_id']],
                    'quality_margin': c['ssim'] - limits[c['image_id']]
                })


if __name__ == '__main__':
    run()
