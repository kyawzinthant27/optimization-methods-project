"""Binary ILP solved with the HiGHS MILP solver through scipy.optimize.milp."""
import math
import time
from collections import Counter


def feasibility(rows, limits):
    qualifying = {
        i: [r for r in rows if r['image_id'] == i and r['ssim'] >= t] for i,
        t in limits.items()
    }
    missing = [i for i, candidates in qualifying.items() if not candidates]
    smallest = {
        i: min(c, key=lambda r: (r['bytes'], r['version']))
        for i,
        c in qualifying.items()
        if c
    }
    highest = {
        i: max(c, key=lambda r: (r['ssim'], -r['bytes'], r['version'])) for i,
        c in qualifying.items() if c
    }
    blockers = [{
        'image_id': i,
        'threshold': limits[i],
        'best_available_ssim': max((r['ssim'] for r in rows if r['image_id'] == i), default=None)
    } for i in missing]

    return {
        'images_without_qualifying_candidate': missing,
        'quality_blockers': blockers,
        'minimum_feasible_bytes': None if missing else sum((r['bytes'] for r in smallest.values())),
        'highest_ssim_bytes': None if missing else sum((r['bytes'] for r in highest.values())),
        'smallest_qualifying_versions': {i: r['version'] for i, r in smallest.items()},
        'highest_ssim_qualifying_versions': {i: r['version'] for i, r in highest.items()}
    }


def checks(selected, limits, budget):
    counts = Counter((r['image_id'] for r in selected))
    total = sum((r['bytes'] for r in selected))
    failures = [r['image_id'] for r in selected if r['ssim'] < limits[r['image_id']]]

    return {
        'one_version_per_image': counts == Counter({i: 1 for i in limits}),
        'budget_ok': total <= budget,
        'quality_ok': not failures,
        'quality_failures': failures,
        'budget_excess_bytes': max(0, total - budget)
    }


def evaluate(selected, limits, budget):
    checked = checks(selected, limits, budget)
    feasible = all((checked[k] for k in ('one_version_per_image', 'budget_ok', 'quality_ok')))

    return {
        'feasible': feasible,
        'total_bytes': sum((r['bytes'] for r in selected)) if selected else None,
        'average_ssim': sum((r['ssim'] for r in selected)) / len(limits) if selected else None,
        'minimum_ssim': min((r['ssim'] for r in selected), default=None),
        'checks': checked,
        'selections': selected
    }


def solve(rows, limits, budget, options=None):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    if type(budget) is not int or budget < 0:
        raise ValueError('Budget must be nonnegative integer bytes')

    preflight = feasibility(rows, limits)
    result = {
        'budget_bytes': budget,
        'preflight': preflight,
        'solver_status': 'not_run',
        'solver_status_code': None,
        'optimality_gap': None,
        'solve_seconds': None,
        'objective_upper_bound': None,
        'solver_node_count': None
    }

    if preflight['images_without_qualifying_candidate'] or budget < preflight['minimum_feasible_bytes']:
        result.update(
            evaluate([], limits, budget),
            status='infeasible_precheck',
            message='No qualifying candidate for some images, or budget below exact feasible minimum'
        )
        return result

    ids = list(limits)
    n = len(rows)
    assignments = np.array([[float(r['image_id'] == i) for r in rows] for i in ids])
    scores = np.array([r['ssim'] for r in rows])
    sizes = np.array([r['bytes'] for r in rows], dtype=float)
    eligible = np.array([float(r['ssim'] >= limits[r['image_id']]) for r in rows])
    constraints = [
        LinearConstraint(assignments, 1, 1),
        LinearConstraint(sizes, -np.inf, budget),
        LinearConstraint(assignments * scores, [limits[i] for i in ids], np.inf)
    ]
    settings = {'presolve': True, 'mip_rel_gap': 0.0, 'time_limit': 60.0, 'disp': False}
    settings.update(options or {})
    start = time.perf_counter()
    raw = milp(
        c=-scores / len(ids),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), eligible),
        constraints=constraints,
        options=settings
    )
    elapsed = time.perf_counter() - start
    selected = [r for r, value in zip(rows, raw.x) if value > 0.5] if raw.x is not None else []
    result.update(evaluate(selected, limits, budget))
    code_names = {
        0: 'optimal',
        1: 'limit_reached',
        2: 'infeasible',
        3: 'unbounded',
        4: 'solver_error'
    }
    status = code_names.get(raw.status, 'unknown')
    integral = raw.x is not None and np.all(np.abs(raw.x - np.rint(raw.x)) <= 1e-06)

    if selected and (not result['feasible'] or not integral):
        status = 'invalid_solver_solution'
        result['feasible'] = False

    if raw.status == 0 and (not result['feasible']):
        status = 'invalid_solver_solution'

    def finite(value):
        return float(value) if value is not None and math.isfinite(value) else None

    bound = finite(getattr(raw, 'mip_dual_bound', None))
    result.update(
        status=status,
        solver_status=code_names.get(raw.status, 'unknown'),
        solver_status_code=int(raw.status),
        message=raw.message,
        optimality_gap=finite(getattr(raw, 'mip_gap', None)),
        solve_seconds=elapsed,
        objective_upper_bound=-bound if bound is not None else None,
        solver_node_count=int(raw.mip_node_count) if getattr(
            raw,
            'mip_node_count',
            None
        ) is not None else None
    )

    return result


def fixed_policy(rows, limits, budget, policy, important):
    base = policy.removeprefix('priority-').removeprefix('all-')
    selected = [
        r
        for r in rows
        if r['version'] == (
            'high'
            if policy.startswith('priority-') and r['image_id'] in important else base
        )
    ]
    result = evaluate(selected, limits, budget)

    return dict(
        result,
        budget_bytes=budget,
        status='feasible' if result['feasible'] else 'infeasible_policy',
        solver_status='not_applicable',
        solver_status_code=None,
        optimality_gap=None,
        solve_seconds=None,
        message='Fixed policy evaluated under identical constraints'
    )
