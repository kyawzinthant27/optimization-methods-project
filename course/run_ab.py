"""Execute only saved Stage A/B cases, preserving all earlier pilot artifacts."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from image_quality.common import read_json, sha256, thresholds, write_json, write_csv
from image_quality.optimize import checks, feasibility, fixed_policy, solve
from image_quality.run import environment, verify_artifacts

PILOTS = ('results/results.json', 'results/sensitivity/results.json')
GENERATION_KEYS = ('image_ids', 'dataset_url', 'max_edge', 'qualities', 'encoder')


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_preserved(inventory):
    for path, expected in inventory['files_sha256'].items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError(f'Preserved pilot file changed: {path}')


def validate_record(row, candidates, limits, budget):
    """Independent checks on a saved/reused record; never trust status alone."""
    lookup = {(r['image_id'], r['version']): r for r in candidates}
    selected = row['selections']

    if row['budget_bytes'] != budget:
        raise ValueError('Result budget mismatch')

    if any((lookup.get((r['image_id'], r['version'])) != r for r in selected)):
        raise ValueError('Selected candidate differs from manifest')

    actual = checks(selected, limits, budget)
    if actual != row['checks']:
        raise ValueError('Constraint checks mismatch')

    feasible = all((actual[k] for k in ('one_version_per_image', 'budget_ok', 'quality_ok')))
    if feasible != row['feasible']:
        raise ValueError('Feasibility flag mismatch')

    if selected:
        if sum((r['bytes'] for r in selected)) != row['total_bytes']:
            raise ValueError('Byte total mismatch')

        for key, value in [
            ('average_ssim', sum((r['ssim'] for r in selected)) / len(limits)),
            ('minimum_ssim', min((r['ssim'] for r in selected)))
        ]:
            if not math.isclose(value, row[key], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f'{key} mismatch')
    elif any((row.get(k) is not None for k in ['total_bytes', 'average_ssim', 'minimum_ssim'])):
        raise ValueError('Empty selection must not have aggregate values')

    if row['status'] == 'optimal' and (not (feasible and row['solver_status_code'] == 0 and (row['optimality_gap'] == 0))):
        raise ValueError('Unsupported optimality claim')

    if row['status'] == 'infeasible_precheck':
        pre = feasibility(candidates, limits)
        if not pre['images_without_qualifying_candidate'] and budget >= pre['minimum_feasible_bytes']:
            raise ValueError('Unsupported infeasibility claim')

        if row['solve_seconds'] is not None:
            raise ValueError('Precheck has fabricated solver time')


def matching_pilot(case, candidates, manifest_hash, current_environment, pilot_sources):
    cfg = case['config']
    limits = thresholds(cfg)

    for path, data in pilot_sources:
        m = data['metadata']
        matches = m['candidate_manifest_sha256'] == manifest_hash and m['thresholds'] == limits and (m['solver_options'] == cfg['solver']) and (m['environment'] == current_environment) and all((
            m['config'][k] == cfg[k]
            for k in GENERATION_KEYS
        ))

        if not matches:
            continue

        for row in data['results']:
            if row['policy'] != 'optimized' or row['budget_bytes'] != case['budget_bytes']:
                continue

            if row['status'] not in ('optimal', 'infeasible_precheck'):
                continue

            validate_record(row, candidates, limits, case['budget_bytes'])
            result = deepcopy(row)
            provenance = {
                'origin': 'reused_pilot',
                'source_path': path,
                'source_sha256': sha256(path),
                'source_scenario': row['scenario'],
                'measured_at_utc': m['measured_at_utc'],
                'runtime_origin': 'original pilot MILP call' if row['solve_seconds'] is not None else 'no MILP call; pilot precheck',
                'matching_fields': [
                    'candidate_manifest_sha256',
                    'generation_settings',
                    'thresholds',
                    'budget_bytes',
                    'solver_options',
                    'environment'
                ],
                'note': 'Reused measurement, not a new independent trial'
            }

            return (result, provenance)

    return None


def enrich(row, case, policy, provenance, candidates):
    row = deepcopy(row)
    limits = thresholds(case['config'])
    pre = feasibility(candidates, limits)
    margins = {r['image_id']: r['ssim'] - limits[r['image_id']] for r in row['selections']}
    reasons = []

    if policy == 'optimized' and (not row['selections']):
        if pre['images_without_qualifying_candidate']:
            reasons.append('No qualifying candidate: ' + ', '.join(pre['images_without_qualifying_candidate']))

        if pre['minimum_feasible_bytes'] is not None and case['budget_bytes'] < pre['minimum_feasible_bytes']:
            reasons.append('Budget below exact feasible minimum')

        if not reasons:
            reasons.append(row.get('message', row['status']))
    else:
        if not row['checks']['one_version_per_image']:
            reasons.append('Not exactly one version per image')

        if not row['checks']['budget_ok']:
            reasons.append(f"Budget exceeded by {row['checks']['budget_excess_bytes']} bytes")

        if row['checks']['quality_failures']:
            reasons.append('Below quality floor: ' + ', '.join(row['checks']['quality_failures']))

    row.update(
        experiment=case['experiment'],
        case_id=case['case_id'],
        policy=policy,
        threshold_0802=limits['0802'],
        thresholds=limits,
        preflight=pre,
        objective=row['average_ssim'] if row['feasible'] else None,
        unused_budget_bytes=case['budget_bytes'] - row['total_bytes'] if row['total_bytes'] is not None else None,
        quality_margins=margins,
        minimum_quality_margin=min(margins.values()) if margins else None,
        qualifying_assignments_before_budget=math.prod((
            sum((r['image_id'] == i and r['ssim'] >= t for r in candidates)) for i,
            t in limits.items()
        )),
        failure_reasons=reasons,
        provenance=provenance
    )
    row.pop('scenario', None)
    row.pop('kind', None)

    return row


SUMMARY_FIELDS = [
    'experiment',
    'case_id',
    'policy',
    'threshold_0802',
    'budget_bytes',
    'feasible',
    'objective',
    'average_ssim',
    'minimum_ssim',
    'total_bytes',
    'unused_budget_bytes',
    'minimum_quality_margin',
    'status',
    'solver_status',
    'solver_status_code',
    'optimality_gap',
    'solve_seconds',
    'origin',
    'measured_at_utc',
    'failure_reasons',
    'selected_versions',
    'quality_margins'
]
SELECTION_FIELDS = [
    'experiment',
    'case_id',
    'policy',
    'feasible',
    'image_id',
    'version',
    'path',
    'bytes',
    'ssim',
    'threshold',
    'quality_margin',
    'sha256'
]


def export_csv(directory, records):
    summaries = []
    selections = []

    for row in records:
        flat = {k: row.get(k) for k in SUMMARY_FIELDS}
        flat.update(
            origin=row['provenance']['origin'],
            measured_at_utc=row['provenance']['measured_at_utc'],
            failure_reasons='; '.join(row['failure_reasons']),
            selected_versions=json.dumps(
                {r['image_id']: r['version'] for r in row['selections']},
                sort_keys=True
            ),
            quality_margins=json.dumps(row['quality_margins'], sort_keys=True)
        )
        summaries.append(flat)

        for r in row['selections']:
            selections.append({
                **r,
                **{k: row[k] for k in ['experiment', 'case_id', 'policy', 'feasible']},
                'threshold': row['thresholds'][r['image_id']],
                'quality_margin': row['quality_margins'][r['image_id']]
            })

    write_csv(directory / 'summary.csv', summaries, SUMMARY_FIELDS)
    write_csv(directory / 'selections.csv', selections, SELECTION_FIELDS)


def run(plan_path, output, reuse=True):
    plan_path = Path(plan_path)
    output = Path(output)

    if output.exists():
        raise FileExistsError(f'Refusing existing output directory: {output}')

    plan = read_json(plan_path)
    inventory_path = 'course/pilots/inventory.json'

    if sha256(inventory_path) != plan['pilot_inventory_sha256']:
        raise ValueError('Pilot inventory hash mismatch')

    inventory = read_json(inventory_path)
    verify_preserved(inventory)

    if sha256('manifests/candidates.json') != plan['candidate_manifest_sha256']:
        raise ValueError('Manifest hash mismatch')

    for key, value in plan['thread_environment'].items():
        if os.environ.get(key) != value:
            raise ValueError(f'Set {key}={value} before starting Python')

    rows = read_json('manifests/candidates.json')
    current = environment()
    cases = []

    for entry in plan['cases']:
        if sha256(entry['config_path']) != entry['sha256']:
            raise ValueError('Case configuration hash mismatch')

        case = read_json(entry['config_path'])
        if (case['experiment'], case['case_id']) != (entry['experiment'], entry['case_id']):
            raise ValueError('Case identity mismatch')

        if case['config']['solver'] != plan['solver_options']:
            raise ValueError('Solver settings changed')

        verify_artifacts(case['config'], rows)
        cases.append(case)

    if len(cases) != plan['planned_case_count']:
        raise ValueError('Incomplete plan')

    sources = [(p, read_json(p)) for p in PILOTS] if reuse else []
    started = now()

    if datetime.fromisoformat(plan['saved_at_utc']) >= datetime.fromisoformat(started):
        raise ValueError('Plan was not saved before execution')

    output.mkdir(parents=True)
    metadata = {
        'started_at_utc': started,
        'plan_path': str(plan_path),
        'plan_sha256': sha256(plan_path),
        'plan_saved_at_utc': plan['saved_at_utc'],
        'environment': current,
        'reuse_enabled': reuse,
        'solver_options': plan['solver_options'],
        'code_sha256': {str(p): sha256(p) for p in [
            Path('course/run_ab.py'),
            Path('image_quality/optimize.py'),
            Path('image_quality/common.py'),
            Path('image_quality/run.py')
        ]},
        'pilot_inventory_sha256': plan['pilot_inventory_sha256'],
        'completed': False
    }
    write_json(output / 'execution.json', metadata)
    records = []

    for case in cases:
        directory = output / case['experiment'] / case['case_id']
        directory.mkdir(parents=True, exist_ok=False)
        limits = thresholds(case['config'])
        case_records = []

        for policy in plan['policies']:
            match = matching_pilot(
                case,
                rows,
                plan['candidate_manifest_sha256'],
                current,
                sources
            ) if policy == 'optimized' else None
            if match:
                result, provenance = match
            else:
                measured_at = now()
                result = solve(
                    rows,
                    limits,
                    case['budget_bytes'],
                    case['config']['solver']
                ) if policy == 'optimized' else fixed_policy(
                    rows,
                    limits,
                    case['budget_bytes'],
                    policy,
                    case['config']['important_thresholds']
                )
                provenance = {
                    'origin': 'new_followup' if policy == 'optimized' else 'fixed_policy_evaluation',
                    'measured_at_utc': measured_at,
                    'runtime_origin': 'new MILP call' if result['solve_seconds'] is not None else 'no MILP call'
                }

            validate_record(result, rows, limits, case['budget_bytes'])
            case_records.append(enrich(result, case, policy, provenance, rows))

        write_json(
            directory / 'results.json',
            {'case': case, 'plan_sha256': metadata['plan_sha256'], 'results': case_records}
        )
        export_csv(directory, case_records)
        records.extend(case_records)
        print(
            f"{case['experiment']}/{case['case_id']}: {case_records[0]['status']} ({case_records[0]['provenance']['origin']})",
            flush=True
        )

    verify_preserved(inventory)
    if sha256(plan_path) != metadata['plan_sha256']:
        raise ValueError('Plan changed during execution')

    for entry in plan['cases']:
        if sha256(entry['config_path']) != entry['sha256']:
            raise ValueError('Case config changed during execution')

    metadata.update(
        finished_at_utc=now(),
        completed=True,
        pilot_files_unchanged=True,
        case_count=len(cases),
        policy_records=len(records),
        new_milp_calls=sum((
            r['policy'] == 'optimized' and r['provenance']['origin'] == 'new_followup' and (r['solve_seconds'] is not None)
            for r in records
        )),
        reused_pilot_records=sum((r['provenance']['origin'] == 'reused_pilot' for r in records))
    )
    write_json(output / 'results.json', {'metadata': metadata, 'results': records})
    export_csv(output, records)
    write_json(output / 'execution.json', metadata)

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='course/plans/stage-ab-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-reuse', action='store_true')
    args = parser.parse_args()
    run(args.plan, args.output, not args.no_reuse)


if __name__ == '__main__':
    main()
