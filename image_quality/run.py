"""Evaluate network scenarios and a measured budget sweep; export results/report."""
import argparse
import os
import platform
import statistics
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from .common import budget_bytes, config, read_json, sha256, thresholds, validate_candidates, write_csv, write_json
from .optimize import feasibility, fixed_policy, solve

POLICIES = ('optimized', 'all-low', 'all-medium', 'all-high', 'priority-low', 'priority-medium')


def scenarios(cfg, preflight):
    result = [{
        'scenario': f'network-{rate}Mbps',
        'kind': 'network',
        'throughput_bits_per_second': int(Decimal(str(rate)) * 1000000),
        'transfer_seconds': cfg['transfer_seconds'],
        'byte_cap': cfg['byte_cap'],
        'budget_bytes': budget_bytes(
            Decimal(str(rate)) * 1000000,
            cfg['transfer_seconds'],
            cfg['byte_cap']
        )
    } for rate in cfg['network_mbps']]
    low, high = (preflight['minimum_feasible_bytes'], preflight['highest_ssim_bytes'])

    if low is not None:
        budgets = sorted({
            low + (high - low) * k // (cfg['sweep_points'] - 1)
            for k in range(cfg['sweep_points'])
        })
        result.extend((
            {'scenario': f'sweep-{k:02d}', 'kind': 'sweep', 'budget_bytes': b} for k,
            b in enumerate(budgets)
        ))

    return result


def verify_artifacts(cfg, rows):
    validate_candidates(rows, cfg['image_ids'])
    generation = read_json('manifests/generation.json')

    for key in ('image_ids', 'dataset_url', 'max_edge', 'qualities', 'encoder'):
        if generation['config'][key] != cfg[key]:
            raise ValueError(f'Generation setting changed ({key}); regenerate candidates first')

    for row in rows:
        if sha256(row['path']) != row['sha256'] or Path(row['path']).stat().st_size != row['bytes']:
            raise ValueError(f"Candidate mismatch: {row['path']}")

    for row in read_json('manifests/references.json'):
        if sha256(row['path']) != row['sha256']:
            raise ValueError(f"Reference mismatch: {row['path']}")


def environment():
    import scipy.optimize._highspy._core as core

    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')

    if cpuinfo.exists():
        cpu = next(
            (
                line.split(':', 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith('model name')
            ),
            cpu
        )

    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'cpu': cpu,
        'logical_cpus': os.cpu_count(),
        'packages': {
            p: version(p)
            for p in ('numpy', 'scipy', 'scikit-image', 'Pillow', 'matplotlib')
        },
        'highs': '.'.join((
            str(getattr(core, f'HIGHS_VERSION_{part}'))
            for part in ('MAJOR', 'MINOR', 'PATCH')
        )),
        'thread_environment': {k: os.environ.get(k) for k in (
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS'
        )}
    }


def export(output, metadata, results):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', {'metadata': metadata, 'results': results})
    summaries, selections = ([], [])

    for row in results:
        summary = {k: v for k, v in row.items() if k not in ('selections', 'checks', 'preflight')}
        summary.update(row['checks'])
        summary['quality_failures'] = ';'.join(row['checks']['quality_failures'])
        summaries.append(summary)

        for selected in row['selections']:
            selections.append(dict(
                selected,
                scenario=row['scenario'],
                policy=row['policy'],
                policy_feasible=row['feasible'],
                budget_bytes=row['budget_bytes'],
                threshold=metadata['thresholds'][selected['image_id']],
                quality_ok=selected['ssim'] >= metadata['thresholds'][selected['image_id']]
            ))

    fields = list(dict.fromkeys((k for row in summaries for k in row)))
    write_csv(output / 'summary.csv', summaries, fields)
    write_csv(
        output / 'selections.csv',
        selections,
        [
            'scenario',
            'policy',
            'policy_feasible',
            'budget_bytes',
            'image_id',
            'version',
            'path',
            'width',
            'height',
            'bytes',
            'ssim',
            'threshold',
            'quality_ok',
            'sha256'
        ]
    )


def plot(output, results, cfg):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')

    for policy in POLICIES:
        points = sorted(
            [
                r
                for r in results
                if r['kind'] == 'sweep' and r['policy'] == policy and r['feasible']
            ],
            key=lambda r: r['budget_bytes']
        )
        if points:
            ax.plot(
                [r['budget_bytes'] / 1000 for r in points],
                [r['average_ssim'] for r in points],
                marker='o' if policy == 'optimized' else '.',
                label=policy,
                linewidth=2 if policy == 'optimized' else 1,
                alpha=0.85
            )

    ids = cfg['image_ids']
    label = 'sensitivity run' if 'experiment_note' in cfg else 'original pilot'
    important = ', '.join((f'{i} ≥ {t}' for i, t in cfg['important_thresholds'].items()))
    ax.set(
        xlabel='Image byte budget (decimal kB = 1,000 bytes)',
        ylabel='Mean RGB SSIM (14 images)'
    )
    ax.set_title(
        f'DIV2K {ids[0]}–{ids[-1]}: {label}\nFeasible policies; important thresholds: {important}',
        fontsize=12
    )
    ax.grid(alpha=0.25)

    if ax.lines:
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            'No feasible sweep: see preflight failures',
            transform=ax.transAxes,
            ha='center'
        )

    fig.savefig(output / 'quality-vs-budget.png', dpi=160)
    fig.savefig(output / 'quality-vs-budget.svg')
    plt.close(fig)


def image_examples(output, rows, results, ids):
    """Predetermined IDs, identical center crops at 1:1 pixels; no cherry-picked crop."""
    from PIL import Image, ImageDraw

    feasible = [
        r
        for r in results
        if r['kind'] == 'sweep' and r['policy'] == 'optimized' and r['feasible']
    ]
    chosen = feasible[len(feasible) // 2] if feasible else None
    examples = []

    for image_id in ids:
        candidates = {r['version']: r for r in rows if r['image_id'] == image_id}
        ref = Image.open(candidates['low']['reference_path']).convert('RGB')
        side = min(256, ref.width, ref.height)
        x, y = ((ref.width - side) // 2, (ref.height - side) // 2)
        box = (x, y, x + side, y + side)
        panels = [('Reference', ref, '')]

        for label in ('low', 'medium', 'high'):
            r = candidates[label]
            panels.append((
                f"{label} q={r['quality']}",
                Image.open(r['path']).convert('RGB'),
                f"{r['bytes']:,} B; SSIM {r['ssim']:.5f}"
            ))

        selected = next(
            (r for r in chosen['selections'] if r['image_id'] == image_id),
            None
        ) if chosen else None
        if selected:
            panels.append((
                f"Selected: {selected['version']}",
                Image.open(selected['path']).convert('RGB'),
                f"{selected['bytes']:,} B; SSIM {selected['ssim']:.5f}"
            ))

        canvas = Image.new('RGB', (len(panels) * (side + 12) + 12, side + 80), 'white')
        draw = ImageDraw.Draw(canvas)

        for index, (title, im, subtitle) in enumerate(panels):
            left = 12 + index * (side + 12)
            draw.text((left, 8), title, fill='black')
            draw.text((left, 28), subtitle, fill='black')
            canvas.paste(im.crop(box), (left, 52))

        path = output / f'example-{image_id}.png'
        canvas.save(path)
        examples.append({
            'image_id': image_id,
            'path': path.name,
            'crop_xyxy': box,
            'selection_scenario': chosen['scenario'] if chosen else None,
            'selection_budget_bytes': chosen['budget_bytes'] if chosen else None
        })

    write_json(output / 'examples.json', examples)
    return examples


def report(output, metadata, results, rows, examples):
    pre = metadata['preflight']
    cfg = metadata['config']
    size_summary = '**No feasible allocation at any budget.** Minimum feasible size and sweep endpoints are undefined.' if pre['minimum_feasible_bytes'] is None else f"Minimum feasible size: **{pre['minimum_feasible_bytes']} bytes**. Sum of highest-SSIM qualifying choices: **{pre['highest_ssim_bytes']} bytes**."
    text = [
        '# Stage 1 measured report',
        '',
        f"Run: {metadata['measured_at_utc']}",
        '',
        cfg.get(
            'experiment_note',
            'Original pilot; thresholds unchanged from the initial configuration.'
        ),
        '',
        '## Experimental assumptions',
        '',
        f"One fixed batch of 14 DIV2K validation images ({', '.join(cfg['image_ids'])}), selected before measuring quality. This convenience batch is not a random or representative sample. Important IDs: " + ', '.join(metadata['config']['important_thresholds']) + '. Thresholds and encoder settings are pilot assumptions, not perceptual guarantees.',
        '',
        f"Thresholds: `{metadata['thresholds']}`.",
        '',
        f"References have a {cfg['max_edge']}-pixel maximum edge (no upscaling), preserving aspect ratio to integer-pixel rounding. Each candidate uses the reference dimensions. WebP quality labels are encoder settings {cfg['qualities']}, not SSIM targets. See generation.json for the actual settings used in this run.",
        '',
        f"Network budgets use floor(bit/s × {cfg['transfer_seconds']} s / 8), optionally limited by a separate byte cap. The allowance is illustrative image transfer time; it excludes latency, HTTP overhead and competing traffic. It is not an FCP/LCP target or browser measurement. The sweep independently spans measured minimum-to-best-quality bytes, without a network cap.",
        '',
        '## Measurements',
        '',
        size_summary,
        '',
        f"Images without a qualifying version: {pre['images_without_qualifying_candidate'] or 'none'}.",
        '',
        f"Quality blockers (measured best versus configured threshold): {pre['quality_blockers'] or 'none'}.",
        '',
        '### Candidate measurements',
        '',
        '| Image | Dimensions | Low bytes / SSIM | Medium bytes / SSIM | High bytes / SSIM | Threshold |',
        '|---|---|---:|---:|---:|---:|'
    ]

    for i in metadata['config']['image_ids']:
        c = {r['version']: r for r in rows if r['image_id'] == i}
        cells = [f"{c[v]['bytes']:,} / {c[v]['ssim']:.6f}" for v in ('low', 'medium', 'high')]
        text.append(f"| {i} | {c['low']['width']}×{c['low']['height']} | " + ' | '.join(cells) + f" | {metadata['thresholds'][i]:.2f} |")

    text += [
        '',
        '### Network scenarios and fixed policies',
        '',
        'Infeasible policy SSIM values describe their files only; they are not valid competitors. Priority-low/medium use high for important images and low/medium for all others.',
        '',
        '| Scenario | Budget B | Policy | Selected B | Mean SSIM | Min SSIM | Status / violations |',
        '|---|---:|---|---:|---:|---:|---|'
    ]
    fmt = lambda value: '—' if value is None else f'{value:.6f}'

    for r in results:
        if r['kind'] != 'network':
            continue

        violations = []
        if r['checks']['budget_excess_bytes']:
            violations.append(f"budget +{r['checks']['budget_excess_bytes']} B")

        if r['checks']['quality_failures']:
            violations.append('quality: ' + ', '.join(r['checks']['quality_failures']))

        if r['status'] == 'infeasible_precheck':
            violations.append('below minimum or missing qualifying image')

        status = r['status'] + ('; ' + '; '.join(violations) if violations else '')
        text.append(f"| {r['scenario']} | {r['budget_bytes']:,} | {r['policy']} | {(
            r['total_bytes']
            if r['total_bytes'] is not None else '—'
        )} | {fmt(r['average_ssim'])} | {fmt(r['minimum_ssim'])} | {status} |")

    text += [
        '',
        '### Budget sweep',
        '',
        '![Quality versus budget](quality-vs-budget.png)',
        '',
        'Only feasible baseline points are plotted. The sampled curve is not an enumeration of all Pareto breakpoints.',
        '',
        '| Budget B | Selected B | Mean SSIM | Min SSIM | Status | Gap | Solver ms | Gain over best feasible fixed policy |',
        '|---:|---:|---:|---:|---|---:|---:|---:|'
    ]
    gains, ties, without_baseline = ([], 0, 0)

    for r in results:
        if r['kind'] != 'sweep' or r['policy'] != 'optimized':
            continue

        baselines = [
            b
            for b in results
            if b['scenario'] == r['scenario'] and b['policy'] != 'optimized' and b['feasible']
        ]
        gain = None

        if baselines and r['feasible']:
            gain = r['average_ssim'] - max((b['average_ssim'] for b in baselines))
            gains.append(gain)
            ties += abs(gain) <= 1e-09
        else:
            without_baseline += 1

        milliseconds = '—' if r['solve_seconds'] is None else f"{r['solve_seconds'] * 1000:.3f}"
        text.append(f"| {r['budget_bytes']:,} | {r['total_bytes']} | {fmt(r['average_ssim'])} | {fmt(r['minimum_ssim'])} | {r['status']} | {r['optimality_gap']} | {milliseconds} | {fmt(gain)} |")

    solved = [r for r in results if r['policy'] == 'optimized' and r['solve_seconds'] is not None]
    if solved:
        times = [r['solve_seconds'] * 1000 for r in solved]
        text += [
            '',
            f'MILP invoked {len(solved)} times. Observed solver-call time: min {min(times):.3f}, median {statistics.median(times):.3f}, max {max(times):.3f} ms. Each scenario is solved once; these are timings across different problems, not repeated-run performance statistics. Timing excludes imports, candidate generation, preflight, model assembly, and report generation; the first call may include solver initialization. No browser was measured.'
        ]

    text += [
        '',
        '## Interpretation and limitations',
        '',
        f'Sweep comparisons tied with the best feasible fixed policy at {ties} points (absolute tolerance 1e-9); {without_baseline} points had no feasible fixed competitor or no optimizer solution.'
    ]
    if gains:
        text += [f'Measured gain over the best feasible fixed policy ranges from {min(gains):.6f} to {max(gains):.6f} SSIM. This is an absolute SSIM difference, not a percentage or a claim of perceptual improvement.']

    text += [
        '',
        'A solver status of optimal means HiGHS established optimality subject to numerical tolerances for this finite candidate set. There is no secondary byte-minimization objective; equivalent-quality selections may differ. A high encoder label is not assumed to have the highest measured SSIM. Results do not generalize beyond this batch, codec, display size, metric, and thresholds. SSIM is measured in gamma-encoded RGB, averaged across channels; it is not a human visual study.',
        '',
        '## Side-by-side examples',
        '',
        f"Predetermined IDs {cfg['image_ids'][0]} and {cfg['image_ids'][2]}; centered 256-pixel crops at 1:1 (or smaller if needed). Reported scores refer to whole images, not crops. The selected column uses the middle feasible sweep point. Examples are illustrative, not a preference test."
    ]

    for e in examples:
        text += [
            '',
            f"Image {e['image_id']}, crop {e['crop_xyxy']}, selected budget {e['selection_budget_bytes']} bytes.",
            '',
            f"![Image {e['image_id']} comparison]({e['path']})"
        ]

    text += [
        '',
        '## Provenance',
        '',
        f"[Official DIV2K dataset and academic-research-only terms](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Original owners retain image copyright. [Dataset citations and method details]({os.path.relpath(
            'README.md',
            output
        )}). SHA-256 originals/references/candidates are in `{os.path.relpath(
            'manifests',
            output
        )}/`. The complete config, package versions, machine details, solver settings, status, bounds, constraints, selections, and elapsed times are in results.json. CSV exports: summary.csv and selections.csv. Stage 2 and Stage 3 have not been run.",
        ''
    ]
    (output / 'report.md').write_text('\n'.join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    cfg = config(args.config)
    rows = read_json('manifests/candidates.json')
    verify_artifacts(cfg, rows)
    limits = thresholds(cfg)
    preflight = feasibility(rows, limits)
    metadata = {
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg,
        'config_sha256': sha256(args.config),
        'candidate_manifest_sha256': sha256('manifests/candidates.json'),
        'thresholds': limits,
        'preflight': preflight,
        'environment': environment(),
        'solver': 'scipy.optimize.milp / bundled HiGHS; binary ILP; deterministic, no application RNG',
        'solver_options': cfg['solver'],
        'scenarios': scenarios(cfg, preflight)
    }
    results = []

    for scenario in metadata['scenarios']:
        for policy in POLICIES:
            result = solve(
                rows,
                limits,
                scenario['budget_bytes'],
                cfg['solver']
            ) if policy == 'optimized' else fixed_policy(
                rows,
                limits,
                scenario['budget_bytes'],
                policy,
                cfg['important_thresholds']
            )
            results.append(dict(
                result,
                **{k: v for k, v in scenario.items() if k != 'budget_bytes'},
                policy=policy
            ))

        print(f"{scenario['scenario']}: {results[-6]['status']}", flush=True)

    export(args.output, metadata, results)
    plot(args.output, results, cfg)
    examples = image_examples(
        args.output,
        rows,
        results,
        [cfg['image_ids'][0], cfg['image_ids'][2]]
    )
    report(args.output, metadata, results, rows, examples)


if __name__ == '__main__':
    main()
