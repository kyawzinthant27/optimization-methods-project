"""Three-cap MILP sensitivity. No image processing or GA calls."""
import argparse
import csv
import itertools
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from image_quality.common import read_json, write_json, sha256, thresholds, budget_bytes
from image_quality.optimize import solve
from image_quality.run import environment


def now():
    return datetime.now(timezone.utc).isoformat()


def independently_check(result, manifest, limits, cap, tolerance):
    chosen = result.get('selections', [])
    if not chosen:
        return {'valid': False, 'reason': 'No returned selection'}

    lookup = {(r['image_id'], r['version']): r for r in manifest}
    matched = [lookup[r['image_id'], r['version']] for r in chosen]
    size = sum((r['bytes'] for r in matched))
    mean = sum((r['ssim'] for r in matched)) / len(limits)
    checks = {
        'one_per_image': Counter((
            r['image_id']
            for r in chosen
        )) == Counter({i: 1 for i in limits}),
        'matches_original_manifest': chosen == matched,
        'all_quality_floors_pass': all((r['ssim'] >= limits[r['image_id']] for r in matched)),
        'integer_byte_sum_matches': size == result['total_bytes'],
        'within_cap': size <= cap,
        'mean_matches': abs(mean - result['average_ssim']) <= tolerance,
        'candidate_files_match': all((
            (ROOT / r['path']).stat().st_size == r['bytes'] and sha256(ROOT / r['path']) == r['sha256']
            for r in matched
        ))
    }

    return dict(
        checks,
        valid=all(checks.values()),
        recomputed_bytes=size,
        recomputed_mean_ssim=mean
    )


def minimize_bytes(rows, limits, cap, optimum, plan):
    """New secondary pass: quality loss no greater than declared match tolerance."""
    n = len(rows)
    assignment = np.array([[float(r['image_id'] == i) for r in rows] for i in limits])
    scores = np.array([r['ssim'] for r in rows])
    sizes = np.array([r['bytes'] for r in rows], dtype=float)
    eligible = np.array([float(r['ssim'] >= limits[r['image_id']]) for r in rows])
    scale = plan['quality_row_scale']
    constraints = [
        LinearConstraint(assignment, 1, 1),
        LinearConstraint(sizes, -np.inf, cap),
        LinearConstraint(assignment * scores, list(limits.values()), np.inf),
        LinearConstraint(
            scores / len(limits) * scale,
            (optimum - plan['match_tolerance']) * scale,
            np.inf
        )
    ]
    begin = time.perf_counter_ns()
    raw = milp(
        c=sizes,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), eligible),
        constraints=constraints,
        options=plan['solver']
    )
    elapsed = (time.perf_counter_ns() - begin) / 1000000.0
    chosen = [r for r, x in zip(rows, raw.x) if x > 0.5] if raw.x is not None else []
    names = {0: 'optimal', 1: 'limit_reached', 2: 'infeasible', 3: 'unbounded', 4: 'solver_error'}
    integral = raw.x is not None and bool(np.all(abs(raw.x - np.rint(raw.x)) <= 1e-06))

    return {
        'status': names.get(raw.status, 'unknown'),
        'solver_status_code': int(raw.status),
        'message': raw.message,
        'optimality_gap': float(raw.mip_gap) if getattr(raw, 'mip_gap', None) is not None else None,
        'solver_ms': elapsed,
        'integral': integral,
        'selections': chosen,
        'total_bytes': sum((r['bytes'] for r in chosen)) if chosen else None,
        'average_ssim': sum((r['ssim'] for r in chosen)) / len(limits) if chosen else None,
        'quality_target': optimum,
        'allowed_quality_loss': plan['match_tolerance']
    }


def audit_reuse(rows, limits, plan):
    archive = read_json(ROOT / 'revisions/user-selected-v1/results/results.json')
    baseline = read_json(ROOT / 'revisions/part1-milp-ga-v1/results/raw.json')
    original_plan = read_json(ROOT / 'revisions/part1-milp-ga-v1/plan.json')
    cfg = read_json(ROOT / 'revisions/user-selected-v1/config.json')
    assert archive['metadata']['manifest_sha256'] == baseline['metadata']['manifest_sha256'] == plan['manifest_sha256']
    assert archive['metadata']['solver_options'] == original_plan['solver'] == plan['solver']
    assert thresholds(cfg) == limits
    assert original_plan['important_ids'] == list(cfg['important_thresholds'])
    assert original_plan['important_floor'] == 0.9 and original_plan['ordinary_floor'] == 0.7
    assert baseline['metadata']['code_sha256']['image_quality/optimize.py'] == sha256(ROOT / 'image_quality/optimize.py')
    sources = [r for r in baseline['runs'] if r['method'] == 'milp' and r['repetition'] == 0]
    sources += [
        r
        for r in archive['results']
        if r['policy'] == 'optimized' and r['thresholds'] == limits
    ]
    reused = {}

    for cap in plan['caps_bytes']:
        for r in sources:
            if r['budget_bytes'] != cap or r['status'] != 'optimal' or r['solver_status_code'] != 0 or (r['optimality_gap'] != 0):
                continue

            check = independently_check(r, rows, limits, cap, plan['match_tolerance'])
            if not check['valid']:
                raise ValueError('Prior claimed optimum failed independent validation')

            source = 'revisions/part1-milp-ga-v1/results/raw.json' if 'method' in r else 'revisions/user-selected-v1/results/results.json'
            reused[cap] = (
                r,
                {
                    'path': source,
                    'sha256': sha256(ROOT / source),
                    'record': {
                        'case_id': r.get('case_id'),
                        'method': r.get('method'),
                        'repetition': r.get('repetition'),
                        'budget_bytes': cap
                    },
                    'manifest_sha256': plan['manifest_sha256'],
                    'solver_settings_match': True,
                    'quality_floors_match': True,
                    'checks': check
                }
            )
            break

    return reused


def verify_enumeration(rows, limits, caps, results, tolerance):
    """Independent finite-space correctness check; not a new algorithm benchmark."""
    domains = [
        [r for r in rows if r['image_id'] == i and r['ssim'] >= floor] for i,
        floor in limits.items()
    ]
    best = {cap: -float('inf') for cap in caps}
    min_tie = {cap: None for cap in caps}
    targets = {
        r['byte_cap']: r['primary']['average_ssim']
        for r in results
        if r['primary'].get('status') == 'optimal'
    }
    count = 0

    for chosen in itertools.product(*domains):
        count += 1
        size = sum((r['bytes'] for r in chosen))

        if size > max(caps):
            continue

        score = sum((r['ssim'] for r in chosen)) / len(limits)
        for cap in caps:
            if size <= cap:
                best[cap] = max(best[cap], score)
                if cap in targets and score >= targets[cap] - tolerance:
                    min_tie[cap] = size if min_tie[cap] is None else min(size, min_tie[cap])

    checks = []

    for r in results:
        cap = r['byte_cap']
        if r['status'] != 'optimal':
            checks.append({'cap': cap, 'checked': False, 'reason': r['status']})
            continue

        assert abs(best[cap] - r['primary']['average_ssim']) <= tolerance
        assert min_tie[cap] == r['bytes_used']
        checks.append({
            'cap': cap,
            'checked': True,
            'exhaustive_best_ssim': best[cap],
            'minimum_bytes_within_quality_tolerance': min_tie[cap]
        })

    return {
        'purpose': 'Independent correctness verification, not timed as solver work',
        'quality_eligible_assignments_examined': count,
        'caps': checks
    }


def run(output):
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing results are preserved')

    plan = read_json(BASE / 'plan.json')

    for p, digest in plan['input_sha256'].items():
        assert sha256(ROOT / p) == digest, p

    rows = read_json(ROOT / plan['manifest'])
    cfg = read_json(ROOT / 'revisions/user-selected-v1/config.json')
    limits = thresholds(cfg)

    for p, digest in read_json(ROOT / 'revisions/user-selected-v1/freeze.json')['files_sha256'].items():
        assert sha256(ROOT / p) == digest, p

    preserved = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in (ROOT / 'revisions').rglob('*')
        if p.is_file() and BASE not in p.parents and ('__pycache__' not in p.parts)
    }
    reused = audit_reuse(rows, limits, plan)
    output.mkdir(parents=True)
    write_json(output / 'preserved-before.json', preserved)
    write_json(output / 'reuse-audit.json', {str(k): v[1] for k, v in reused.items()})
    write_json(
        output / 'execution.json',
        {
            'started_at_utc': now(),
            'plan_sha256': sha256(BASE / 'plan.json'),
            'environment': environment(),
            'runner_sha256': sha256(Path(__file__)),
            'ga_calls': 0
        }
    )
    results = []

    for target, cap in zip(plan['target_seconds'], plan['caps_bytes']):
        assert budget_bytes(4000000, target) == cap
        if cap in reused:
            primary, provenance = reused[cap]
            origin, current_ms = ('reused_proven_optimum', None)
        else:
            begin = time.perf_counter_ns()
            primary = solve(rows, limits, cap, plan['solver'])
            current_ms = (time.perf_counter_ns() - begin) / 1000000.0
            provenance, origin = (None, 'fresh_existing_MILP')

        record = {
            'target_seconds': target,
            'byte_cap': cap,
            'primary': primary,
            'primary_origin': origin,
            'primary_source': provenance,
            'primary_new_call_ms': current_ms,
            'primary_saved_solver_ms': primary.get(
                'solve_seconds',
                0
            ) * 1000 if primary.get('solve_seconds') is not None else None,
            'status': primary['status'],
            'bytes_used': None,
            'optimal_mean_ssim': None,
            'selected_mean_ssim': None,
            'unused_bytes': None,
            'selections': []
        }
        if primary['status'] == 'optimal' and primary['optimality_gap'] == 0:
            check = independently_check(primary, rows, limits, cap, plan['match_tolerance'])
            assert check['valid'], check
            record['primary_checks'] = check
            secondary = minimize_bytes(rows, limits, cap, primary['average_ssim'], plan)
            record['secondary'] = secondary
            record['status'] = 'tie_break_' + secondary['status']

            if secondary['status'] == 'optimal' and secondary['optimality_gap'] == 0:
                check = independently_check(secondary, rows, limits, cap, plan['match_tolerance'])
                quality_loss = primary['average_ssim'] - secondary['average_ssim']
                check['preserves_optimal_quality'] = abs(quality_loss) <= plan['match_tolerance']
                check['integral'] = secondary['integral']
                record['secondary_checks'] = check

                if check['valid'] and check['preserves_optimal_quality'] and check['integral']:
                    record.update(
                        status='optimal',
                        bytes_used=secondary['total_bytes'],
                        optimal_mean_ssim=primary['average_ssim'],
                        selected_mean_ssim=secondary['average_ssim'],
                        unused_bytes=cap - secondary['total_bytes'],
                        quality_loss_in_tiebreak=quality_loss,
                        selections=secondary['selections']
                    )
                else:
                    record['status'] = 'invalid_secondary_solution'

        results.append(record)
        write_json(output / f'{cap}.json', record)

    ref = next((r for r in results if r['byte_cap'] == 1250000))

    for r in results:
        r['ssim_loss_vs_2_50'] = ref['optimal_mean_ssim'] - r['optimal_mean_ssim'] if ref['status'] == r['status'] == 'optimal' else None
        reference = {s['image_id']: s for s in ref['selections']}
        r['changes_vs_2_50'] = [{
            'image_id': c['image_id'],
            'from_version': reference[c['image_id']]['version'],
            'to_version': c['version'],
            'from_file': reference[c['image_id']]['path'],
            'to_file': c['path'],
            'bytes_difference': c['bytes'] - reference[c['image_id']]['bytes'],
            'ssim_difference': c['ssim'] - reference[c['image_id']]['ssim']
        } for c in r['selections'] if c['image_id'] in reference and c['version'] != reference[c['image_id']]['version']]

    exhaustive = verify_enumeration(
        rows,
        limits,
        plan['caps_bytes'],
        results,
        plan['match_tolerance']
    )

    for p, digest in preserved.items():
        assert sha256(ROOT / p) == digest, p

    payload = {
        'plan_sha256': sha256(BASE / 'plan.json'),
        'manifest_sha256': plan['manifest_sha256'],
        'thresholds': limits,
        'match_tolerance': plan['match_tolerance'],
        'results': results
    }
    write_json(output / 'results.json', payload)
    columns = [
        'target_seconds',
        'byte_cap',
        'status',
        'bytes_used',
        'unused_bytes',
        'optimal_mean_ssim',
        'selected_mean_ssim',
        'ssim_loss_vs_2_50',
        'quality_loss_in_tiebreak',
        'primary_origin',
        'primary_new_call_ms',
        'primary_saved_solver_ms'
    ]

    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    with (output / 'selections.csv').open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            ['target_seconds', 'image_id', 'version', 'path', 'bytes', 'ssim', 'threshold'],
            extrasaction='ignore'
        )
        writer.writeheader()

        for r in results:
            for c in r['selections']:
                writer.writerow(dict(c, target_seconds=r['target_seconds']))

    write_json(
        output / 'verification.json',
        {
            'passed': True,
            'checked_at_utc': now(),
            'preserved_files_unchanged': len(preserved),
            'exhaustive_verification': exhaustive,
            'ga_calls': 0,
            'new_primary_solves': sum((
                r['primary_origin'] == 'fresh_existing_MILP'
                for r in results
            )),
            'new_secondary_solves': sum(('secondary' in r for r in results))
        }
    )
    lines = [
        '# MILP byte-budget sensitivity',
        '',
        'Same 14 images, full-precision measured SSIM, 0.90/0.70 floors and mean-SSIM objective. 4 Mbps, decimal units. These are modeled payload-transfer allowances, not measured page-load times.',
        '',
        '| Target (s) | Cap (bytes) | Status | Used bytes | Optimal mean SSIM | Loss vs 2.50 s |',
        '|---|---|---|---|---|---|'
    ]

    for r in results:
        lines.append(f"| {r['target_seconds']} | {r['byte_cap']} | {r['status']} | {r['bytes_used']} | {r['optimal_mean_ssim']} | {r['ssim_loss_vs_2_50']} |")

    lines += [
        '',
        'The quality optimum was reused for 1,000,000 and 1,250,000 bytes after exact-input and proven-optimal checks. The 1,125,000-byte quality solve is new. All three byte-minimizing second passes are new. The checkout contains no implemented epsilon-budget sweep or MILP secondary tie-break; this revision adds the requested secondary pass without altering the original solver. Quality is preserved within the existing absolute mean-SSIM match tolerance of 1e-10. The unrounded SSIM row is scaled by 1e6 for numerical conditioning. This is a secondary tie-break, not a new weighted objective.',
        '',
        '## Returned version changes relative to 2.50 seconds'
    ]

    for r in results:
        changes = ', '.join((
            f"{c['image_id']}: {c['from_version']} → {c['to_version']}"
            for c in r['changes_vs_2_50']
        )) or 'None'
        lines.append(f"- {r['target_seconds']} s: {changes}.")

    lines += [
        '',
        'These describe the returned solutions, not changes that every tied optimum must make. Uniqueness is not claimed.',
        '',
        '## Why re-solve rather than use a shadow price?',
        'The image choices are discrete. As the byte cap grows, the best SSIM can remain flat until another combination becomes feasible, then jump. An LP-relaxation dual describes a continuous relaxation and is not an exact sensitivity price for this binary model. We therefore evaluate each exact cap with a proven MILP optimum, reusing only an exactly matching certified result.',
        '',
        'Both solver passes retain status, reported gap and runtime in results.json. Reused solver timings are historical, not new measurements. Fresh call-to-return timing and secondary solver-only timing have separate fields; no timing benchmark or browser experiment was run.',
        '',
        f"Independent enumeration checked all {exhaustive['quality_eligible_assignments_examined']:,} quality-eligible assignments and verified the quality maxima and minimum tie-break byte totals. {len(preserved)} existing revision files remain unchanged. No GA calls were made."
    ]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(
        [{k: r[k] for k in [
            'target_seconds',
            'status',
            'bytes_used',
            'optimal_mean_ssim',
            'ssim_loss_vs_2_50',
            'changes_vs_2_50'
        ]} for r in results],
        indent=2
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE / 'results')
    run(parser.parse_args().output)
