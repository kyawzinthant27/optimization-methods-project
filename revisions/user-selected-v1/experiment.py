"""Confirmed DIV2K batch. Separate outputs; reuse the existing scientific functions."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import argparse, csv, json, math, os, platform, shutil, sys, urllib.request, zipfile, zlib

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
from image_quality.common import read_json, sha256, write_csv, write_json, budget_bytes, thresholds, validate_candidates
from image_quality.download import RangeReader

URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
SOURCE = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/'
SELECTION = ROOT / 'image-selector/selections/selection-2026-11-04T06-11-34-432Z-1fedc31e.json'
IDS = [
    '0801',
    '0803',
    '0807',
    '0808',
    '0810',
    '0811',
    '0814',
    '0823',
    '0826',
    '0829',
    '0833',
    '0842',
    '0851',
    '0853'
]
IMPORTANT = ['0801', '0803', '0829', '0853']

THREADS = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
POLICIES = ['optimized', 'all-low', 'all-medium', 'all-high', 'priority-low', 'priority-medium']
now = lambda: datetime.now(timezone.utc).isoformat()
rel = lambda p: str(Path(p).resolve().relative_to(ROOT))


def load(name):
    return read_json(BASE / name)


def save(name, value):
    write_json(BASE / name, value)


def newdir(name):
    p = BASE / name
    p.mkdir(parents=True, exist_ok=False)

    return p


def environment():
    import numpy, scipy
    from scipy.optimize._highspy._core import HIGHS_VERSION_MAJOR, HIGHS_VERSION_MINOR, HIGHS_VERSION_PATCH
    from PIL import features

    cpu = next(
        (
            l.split(':', 1)[1].strip()
            for l in Path('/proc/cpuinfo').read_text().splitlines()
            if l.startswith('model name')
        ),
        'unknown'
    )
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'cpu': cpu,
        'logical_cpus': os.cpu_count(),
        'packages': {p: version(p) for p in ['Pillow', 'numpy', 'scipy', 'scikit-image']},
        'libwebp': features.version('webp'),
        'littlecms': features.version('littlecms2'),
        'highs': f'{HIGHS_VERSION_MAJOR}.{HIGHS_VERSION_MINOR}.{HIGHS_VERSION_PATCH}',
        'thread_environment': {k: os.environ.get(k) for k in THREADS}
    }


def preserve():
    inventory = load('preserved-before.json')

    for p, h in inventory.items():
        if sha256(ROOT / p) != h:
            raise ValueError(f'Existing artifact changed: {p}')

    return len(inventory)


def init():
    if (BASE / 'plan.json').exists():
        raise FileExistsError('Plan exists; use a separate directory for a new run.')

    selected = read_json(SELECTION)
    assert [i['source_image_id'] for i in selected['images']] == IDS
    assert [i['source_image_id'] for i in selected['images'] if i['important']] == IMPORTANT
    assert all((i['minimum_ssim'] == (0.9 if i['important'] else 0.7) for i in selected['images']))
    cfg = read_json(ROOT / 'config.json')
    cfg.update(
        image_ids=IDS,
        ordinary_threshold=0.7,
        important_thresholds={i: 0.9 for i in IMPORTANT}
    )
    cfg['transfer_seconds'] = '2.5'
    save('config.json', cfg)
    save('selection-draft-copy.json', selected)
    save(
        'confirmation.json',
        {
            'recorded_at_utc': now(),
            'source_draft': rel(SELECTION),
            'draft_sha256': sha256(SELECTION),
            'authorization': 'User explicitly confirmed these exact 14 images and four featured assignments and requested Stage 2, including obtaining the seven missing HR originals.',
            'image_ids': IDS,
            'important_ids': IMPORTANT,
            'thresholds': thresholds(cfg),
            'processing_authorized': True
        }
    )
    files = []

    for folder in [
        'data',
        'manifests',
        'results',
        'reports',
        'course',
        'image_quality',
        'revisions/candidate-preparation-v1',
        'revisions/network-time-v1'
    ]:
        files += [
            p
            for p in (ROOT / folder).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts
        ]

    files += [ROOT / p for p in [
        'config.json',
        'config-sensitivity.json',
        'requirements.lock.txt',
        'requirements.txt'
    ]] + [SELECTION]
    save('preserved-before.json', {rel(p): sha256(p) for p in sorted(set(files))})
    earlier = read_json(ROOT / 'revisions/network-time-v1/plan.json')
    cases = [{
        'case_id': f'{mbps}Mbps-{seconds}s',
        'kind': 'network_time',
        'throughput_mbps': mbps,
        'transfer_seconds': seconds,
        'budget_bytes': budget_bytes(mbps * 1000000, seconds)
    } for mbps in earlier['network_mbps'] for seconds in earlier['allowances_seconds']]
    save(
        'plan.json',
        {
            'saved_at_utc': now(),
            'label': 'User-selected DIV2K batch v1; academic e-commerce-style importance roles applied to general photographs',
            'confirmed_selection_sha256': sha256(BASE / 'confirmation.json'),
            'config_sha256': sha256(BASE / 'config.json'),
            'image_ids': IDS,
            'important_ids': IMPORTANT,
            'thresholds': thresholds(cfg),
            'generation_rule': 'q40 / q70 / q95, lossy WebP method=6 exact=True, no metadata. No q98 exception or adaptive quality escalation. Stop if an image has no qualifying candidate.',
            'method': 'Reuse image_quality.generate.canonical_image and measure unchanged. Opaque RGB sources; EXIF orientation and ICC-to-sRGB; single LANCZOS resize, max edge 960, no upscaling; gamma-encoded RGB uint8 SSIM.',
            'source_policy': 'Only official DIV2K high-resolution archive members or checksum-verified existing HR originals. Never selection previews. Academic research only; owner attribution is not individually supplied by the dataset.',
            'cases': cases,
            'extra_boundary_cases': 'After measurement test exact minimum bytes minus one, at minimum, and plus one. These are numerical tests, not network-precision claims.',
            'policies': POLICIES,
            'solver_options': cfg['solver'],
            'thread_environment': THREADS,
            'nominal_assignments': 3 ** 14,
            'assignment_count_rule': 'Quality-only count is the product of eligible version counts. Do not call either count the budget-feasible count.',
            'budget_assumptions': 'Reuse all 21 prior network/time scenarios; B=floor(Mbps*1000000*seconds/8). No byte cap, no relaxed thresholds. Transfer estimates exclude latency, overhead, competing traffic, decode and paint.',
            'output_policy': 'New directory only; fresh measurements and solves; freeze inputs before optimization; no GA, multi-objective extension or new browser benchmarking.'
        }
    )
    print('Saved confirmed selection, configuration, plan and preservation inventory.', flush=True)


def inspect():
    if (BASE / 'download-plan.json').exists():
        raise FileExistsError('Download plan exists.')

    with urllib.request.urlopen(SOURCE, timeout=30) as response:
        html = response.read().decode()

    if 'academic research purpose only' not in html:
        raise ValueError('Official usage terms need review; stopping.')

    (BASE / 'official-source.html').write_text(html)
    old = {i['image_id']: i for i in read_json(ROOT / 'manifests/originals.json')['images']}
    missing = [i for i in IDS if i not in old]
    reader = RangeReader(URL, max_bytes=200000)

    with zipfile.ZipFile(reader) as archive:
        members = [archive.getinfo(f'DIV2K_valid_HR/{i}.png') for i in IDS]
        records = [
            {
                'image_id': i,
                'zip_member': m.filename,
                'zip_crc32': f'{m.CRC:08x}',
                'uncompressed_bytes': m.file_size,
                'compressed_bytes': m.compress_size,
                'obtain': 'download' if i in missing else 'verified_local_copy'
            } for i,
            m in zip(IDS, members)
        ]
        transfer = sum((
            r['compressed_bytes']
            for r in records
            if r['obtain'] == 'download'
        )) + 200000

    save(
        'source-terms.json',
        {
            'verified_at_utc': now(),
            'official_page': SOURCE,
            'archive_url': URL,
            'local_page_snapshot': 'official-source.html',
            'terms': 'Academic research purposes only. Copyright belongs to original image owners. Individual original-owner names and individual licenses are not listed on the dataset page.',
            'permission_basis': 'Published DIV2K academic-use terms for this CPE 613 research example; no commercial-use permission claimed.',
            'dataset_citations': [
                'Agustsson and Timofte, NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study, CVPR Workshops, 2017.',
                'Timofte et al., NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results, CVPR Workshops, 2017.'
            ]
        }
    )
    save(
        'download-plan.json',
        {
            'saved_at_utc': now(),
            'archive_url': URL,
            'archive_etag': reader.etag,
            'archive_last_modified': reader.modified,
            'missing_ids': missing,
            'estimated_transfer_bytes': transfer,
            'members': records
        }
    )
    print(
        json.dumps({
            'missing_ids': missing,
            'estimated_transfer_bytes': transfer,
            'full_archive_bytes': reader.size
        }),
        flush=True
    )


def acquire():
    if (BASE / 'originals.json').exists():
        raise FileExistsError('Original manifest exists.')

    plan = load('download-plan.json')
    folder = BASE / 'originals'
    folder.mkdir(exist_ok=True)
    old = {i['image_id']: i for i in read_json(ROOT / 'manifests/originals.json')['images']}
    cfg = load('config.json')
    limits = thresholds(cfg)
    reader = RangeReader(URL, max_bytes=plan['estimated_transfer_bytes'])

    if reader.etag != plan['archive_etag']:
        raise ValueError('Official archive changed after planning; stop for review.')

    records = []
    current = None

    try:
        with zipfile.ZipFile(reader) as archive:
            for entry in plan['members']:
                current = i = entry['image_id']
                member = archive.getinfo(entry['zip_member'])
                dest = folder / f'{i}.png'
                assert member.file_size == entry['uncompressed_bytes'] and f'{member.CRC:08x}' == entry['zip_crc32']
                if not dest.exists():
                    if entry['obtain'] == 'verified_local_copy':
                        source = ROOT / old[i]['path']
                        assert sha256(source) == old[i]['sha256']
                        shutil.copyfile(source, dest)
                    else:
                        dest.write_bytes(archive.read(member))

                data = dest.read_bytes()
                if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
                    raise ValueError(f'Official archive verification failed for {i}')

                records.append({
                    'image_id': i,
                    'path': rel(dest),
                    'sha256': sha256(dest),
                    'bytes': len(data),
                    'source_page': SOURCE,
                    'dataset_url': URL,
                    'zip_member': member.filename,
                    'zip_crc32': entry['zip_crc32'],
                    'acquisition': entry['obtain'],
                    'important': i in IMPORTANT,
                    'threshold': limits[i],
                    'attribution': 'DIV2K dataset; copyright belongs to the original image owners; individual owner not identified by dataset.',
                    'permission': 'Academic research only, per official DIV2K source page',
                    'source_terms_path': rel(BASE / 'source-terms.json')
                })
                print(f'Obtained and verified HR original {i}', flush=True)
    except Exception as exc:
        save(
            'acquisition-failure.json',
            {'image_id': current, 'error': str(exc), 'stopped_at_utc': now()}
        )
        raise

    save(
        'originals.json',
        {
            'obtained_at_utc': now(),
            'archive_url': URL,
            'archive_etag': reader.etag,
            'transferred_bytes': reader.transferred,
            'images': records
        }
    )
    preserve()
    print('All 14 HR originals verified; seven local copies and seven new downloads.', flush=True)


def prepare():
    from PIL import Image
    from image_quality.generate import canonical_image, measure, SSIM_SETTINGS
    from image_quality.optimize import feasibility

    cfg = load('config.json')
    limits = thresholds(cfg)
    sources = load('originals.json')['images']
    assert [s['image_id'] for s in sources] == IDS

    for source in sources:
        p = ROOT / source['path']
        assert sha256(p) == source['sha256']
        with Image.open(p) as im:
            if im.mode != 'RGB':
                raise ValueError(f"{source['image_id']} is {im.mode}; methodological review required before conversion.")

    refsdir = newdir('references')
    candidatesdir = newdir('candidates')
    rows = []
    refs = []

    for source in sources:
        i = source['image_id']
        reference, original_dimensions, color = canonical_image(
            ROOT / source['path'],
            cfg['max_edge']
        )
        refpath = refsdir / f'{i}.png'
        reference.save(refpath, format='PNG', optimize=False, compress_level=9)
        refs.append({
            'image_id': i,
            'path': rel(refpath),
            'sha256': sha256(refpath),
            'width': reference.width,
            'height': reference.height,
            'original_dimensions': original_dimensions,
            'color_handling': color,
            'source_sha256': source['sha256']
        })

        for label, quality in cfg['qualities'].items():
            target = candidatesdir / f'{i}-{label}.webp'
            reference.save(
                target,
                format='WEBP',
                quality=quality,
                **cfg['encoder'],
                icc_profile=b'',
                exif=b'',
                xmp=b''
            )

            with Image.open(target) as decoded:
                assert decoded.size == reference.size
                score = measure(reference, decoded.convert('RGB'))

            rows.append({
                'image_id': i,
                'version': label,
                'path': rel(target),
                'width': reference.width,
                'height': reference.height,
                'bytes': target.stat().st_size,
                'ssim': score,
                'quality': quality,
                'lossless': False,
                'sha256': sha256(target),
                'reference_path': rel(refpath),
                'original_path': source['path'],
                'source_sha256': source['sha256'],
                'important': i in IMPORTANT,
                'threshold': limits[i],
                'quality_eligible': score >= limits[i],
                'quality_margin': score - limits[i],
                'source_page': SOURCE,
                'dataset_url': URL,
                'zip_member': source['zip_member'],
                'attribution': source['attribution'],
                'permission': source['permission']
            })

        print('Measured three versions:', i, flush=True)

    validate_candidates(rows, IDS)
    pre = feasibility(rows, limits)
    per_image = [{
        'image_id': i,
        'important': i in IMPORTANT,
        'threshold': limits[i],
        'generated': sum((r['image_id'] == i for r in rows)),
        'eligible': sum((r['image_id'] == i and r['quality_eligible'] for r in rows)),
        'best_ssim': max((r['ssim'] for r in rows if r['image_id'] == i))
    } for i in IDS]
    save('manifest.json', rows)
    write_csv(BASE / 'manifest.csv', rows, list(rows[0]))
    save('references.json', refs)
    save(
        'generation.json',
        {
            'started_from': '14 verified HR originals; no selection preview used',
            'completed_at_utc': now(),
            'config': cfg,
            'ssim_settings': SSIM_SETTINGS,
            'environment': environment(),
            'method_change': False,
            'code_sha256': {p: sha256(ROOT / p) for p in [
                'image_quality/generate.py',
                'image_quality/common.py',
                'image_quality/optimize.py'
            ]}
        }
    )
    save(
        'feasibility.json',
        {
            'checked_at_utc': now(),
            'preflight': pre,
            'per_image': per_image,
            'nominal_assignments': 3 ** 14,
            'quality_only_assignments': math.prod((r['eligible'] for r in per_image)),
            'budget_feasible_assignments': 'Not enumerated; depends on scenario budget. Feasibility is determined independently for every scenario.',
            'course_requirement': 'Nominal count exceeds 3,000,000. Whether the course requires nominal or feasible assignments is not independently verified; do not equate these counts.'
        }
    )
    preserve()
    print(json.dumps(load('feasibility.json')), flush=True)

    if pre['images_without_qualifying_candidate']:
        raise SystemExit('STOP: no eligible candidate for ' + ', '.join(pre['images_without_qualifying_candidate']))


def verify_dataset():
    from PIL import Image
    from image_quality.generate import measure
    from image_quality.optimize import feasibility

    cfg = load('config.json')
    limits = thresholds(cfg)
    rows = load('manifest.json')
    originals = load('originals.json')['images']
    refs = load('references.json')
    assert cfg['image_ids'] == IDS and list(cfg['important_thresholds']) == IMPORTANT
    assert len(IMPORTANT) == 4 and len(set(IDS)) == 14
    assert all((limits[i] == (0.9 if i in IMPORTANT else 0.7) for i in IDS))
    validate_candidates(rows, IDS)

    for source in originals:
        p = ROOT / source['path']
        assert sha256(p) == source['sha256'] and p.stat().st_size == source['bytes']
        assert zlib.crc32(p.read_bytes()) == int(source['zip_crc32'], 16)

    for ref in refs:
        assert sha256(ROOT / ref['path']) == ref['sha256']
        with Image.open(ROOT / ref['path']) as reference:
            for row in [r for r in rows if r['image_id'] == ref['image_id']]:
                p = ROOT / row['path']
                assert sha256(p) == row['sha256'] and p.stat().st_size == row['bytes']
                with Image.open(p) as candidate:
                    assert candidate.size == reference.size == (row['width'], row['height'])
                    assert abs(measure(reference, candidate.convert('RGB')) - row['ssim']) <= 1e-12

                assert row['quality_eligible'] == (row['ssim'] >= limits[row['image_id']])
                assert row['quality'] == cfg['qualities'][row['version']]

    assert feasibility(rows, limits) == load('feasibility.json')['preflight']
    return {
        'checked_at_utc': now(),
        'status': 'passed',
        'originals': 14,
        'candidates': 42,
        'important_images': 4,
        'remeasured_candidate_ssim': 42,
        'old_files_unchanged': preserve()
    }


def freeze():
    if (BASE / 'freeze.json').exists():
        raise FileExistsError('Freeze exists.')

    check = verify_dataset()
    save('dataset-validation.json', check)
    pre = load('feasibility.json')['preflight']

    if pre['images_without_qualifying_candidate']:
        raise ValueError('Unqualified image; no solving permitted.')

    paths = [BASE / p for p in [
        'plan.json',
        'config.json',
        'confirmation.json',
        'originals.json',
        'references.json',
        'manifest.json',
        'manifest.csv',
        'generation.json',
        'feasibility.json',
        'source-terms.json',
        'official-source.html',
        'experiment.py'
    ]]
    paths += [
        ROOT / r['path']
        for r in load('originals.json')['images'] + load('references.json') + load('manifest.json')
    ]
    save(
        'freeze.json',
        {
            'frozen_at_utc': now(),
            'manifest_sha256': sha256(BASE / 'manifest.json'),
            'files_sha256': {rel(p): sha256(p) for p in paths}
        }
    )
    print('Dataset validated and frozen before optimization.', flush=True)


def verify_frozen():
    for p, h in load('freeze.json')['files_sha256'].items():
        assert sha256(ROOT / p) == h, f'Frozen input changed: {p}'


def run():
    from image_quality.optimize import solve, fixed_policy, checks
    from course.run_ab import validate_record

    verify_frozen()
    preserve()
    cfg = load('config.json')
    limits = thresholds(cfg)
    rows = load('manifest.json')
    pre = load('feasibility.json')['preflight']
    assert not pre['images_without_qualifying_candidate']

    for key, value in THREADS.items():
        assert os.environ.get(key) == value

    cases = load('plan.json')['cases']
    lo = pre['minimum_feasible_bytes']
    cases += [
        {
            'case_id': label,
            'kind': 'byte_boundary',
            'budget_bytes': lo + offset,
            'throughput_mbps': None,
            'transfer_seconds': None
        } for label,
        offset in [('minimum-minus-one', -1), ('minimum', 0), ('minimum-plus-one', 1)]
    ]
    casefolder = newdir('configs')
    entries = []

    for case in cases:
        p = casefolder / (case['case_id'] + '.json')
        write_json(p, {**case, 'thresholds': limits, 'solver_options': cfg['solver']})
        entries.append({'path': rel(p), 'sha256': sha256(p)})

    save(
        'solve-plan.json',
        {
            'saved_at_utc': now(),
            'freeze_sha256': sha256(BASE / 'freeze.json'),
            'manifest_sha256': sha256(BASE / 'manifest.json'),
            'cases': entries,
            'policies': POLICIES,
            'fresh_solves_only': True
        }
    )
    output = newdir('results')
    started = now()
    records = []

    for case in cases:
        case_records = []
        for policy in POLICIES:
            result = solve(
                rows,
                limits,
                case['budget_bytes'],
                cfg['solver']
            ) if policy == 'optimized' else fixed_policy(
                rows,
                limits,
                case['budget_bytes'],
                policy,
                IMPORTANT
            )
            validate_record(result, rows, limits, case['budget_bytes'])
            failures = []

            if result['status'] == 'infeasible_precheck':
                failures = ['Budget below minimum qualifying batch']
            else:
                if not result['checks']['budget_ok']:
                    failures.append('Budget exceeded')

                if result['checks']['quality_failures']:
                    failures.append('Quality floor failed: ' + ', '.join(result['checks']['quality_failures']))

            result.update(
                **case,
                policy=policy,
                thresholds=limits,
                objective=result['average_ssim'] if result['feasible'] else None,
                unused_budget_bytes=case['budget_bytes'] - result['total_bytes'] if result['total_bytes'] is not None else None,
                quality_margins={
                    r['image_id']: r['ssim'] - limits[r['image_id']]
                    for r in result['selections']
                },
                failure_reasons=failures,
                measured_at_utc=now()
            )
            case_records.append(result)
            records.append(result)

        write_json(output / (case['case_id'] + '.json'), case_records)
        print(case['case_id'], case_records[0]['status'], flush=True)

    summary_fields = [
        'case_id',
        'kind',
        'throughput_mbps',
        'transfer_seconds',
        'policy',
        'budget_bytes',
        'feasible',
        'objective',
        'total_bytes',
        'unused_budget_bytes',
        'average_ssim',
        'minimum_ssim',
        'status',
        'solver_status',
        'solver_status_code',
        'optimality_gap',
        'solve_seconds',
        'selected_versions',
        'quality_margins',
        'failure_reasons'
    ]
    summaries = [{
        **{k: r.get(k) for k in summary_fields},
        'selected_versions': json.dumps({s['image_id']: s['version'] for s in r['selections']}),
        'quality_margins': json.dumps(r['quality_margins']),
        'failure_reasons': '; '.join(r['failure_reasons'])
    } for r in records]
    write_csv(output / 'summary.csv', summaries, summary_fields)
    selections = [{
        'case_id': r['case_id'],
        'policy': r['policy'],
        'feasible': r['feasible'],
        **{k: s[k] for k in [
            'image_id',
            'version',
            'path',
            'bytes',
            'ssim',
            'threshold',
            'quality_margin',
            'sha256'
        ]}
    } for r in records for s in r['selections']]
    write_csv(output / 'selections.csv', selections, list(selections[0]))
    verify_frozen()
    save(
        'results/results.json',
        {
            'metadata': {
                'started_at_utc': started,
                'completed_at_utc': now(),
                'environment': environment(),
                'manifest_sha256': sha256(BASE / 'manifest.json'),
                'solve_plan_sha256': sha256(BASE / 'solve-plan.json'),
                'freeze_sha256': sha256(BASE / 'freeze.json'),
                'solver_options': cfg['solver'],
                'old_files_unchanged': preserve(),
                'reused_results': 0,
                'timing': 'One fresh call per feasible case; MILP call only; excludes preprocessing, checks, model assembly and page loading.'
            },
            'results': records
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['init', 'inspect', 'acquire', 'prepare', 'freeze', 'run'])
    args = parser.parse_args()
    os.chdir(ROOT)
    globals()[args.stage]()
