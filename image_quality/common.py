"""Small serialization, validation and units helpers; no scientific imports."""
import csv
import hashlib
import json
import math
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def config(path='config.json'):
    cfg = read_json(path)
    ids = cfg['image_ids']

    if len(ids) != 14 or len(set(ids)) != 14:
        raise ValueError('This experiment requires exactly 14 distinct image IDs')

    if any((len(i) != 4 or not i.isdecimal() or (not 801 <= int(i) <= 900) for i in ids)):
        raise ValueError('Expected four-digit DIV2K validation IDs 0801–0900')

    if set(cfg['qualities']) != {'low', 'medium', 'high'}:
        raise ValueError('Exactly low, medium and high versions are required')

    if any((type(q) is not int or not 0 <= q <= 100 for q in cfg['qualities'].values())):
        raise ValueError('WebP quality must be an integer in [0, 100]')

    if type(cfg['max_edge']) is not int or cfg['max_edge'] < 11:
        raise ValueError('max_edge must be an integer >= 11')

    if not set(cfg['important_thresholds']).issubset(ids):
        raise ValueError('Important images must belong to the batch')

    thresholds(cfg)
    if type(cfg['sweep_points']) is not int or cfg['sweep_points'] < 2:
        raise ValueError('At least two sweep points required')

    for mbps in cfg['network_mbps']:
        budget_bytes(Decimal(str(mbps)) * 1000000, cfg['transfer_seconds'], cfg['byte_cap'])

    return cfg


def thresholds(cfg):
    result = {
        i: cfg['important_thresholds'].get(i, cfg['ordinary_threshold'])
        for i in cfg['image_ids']
    }
    if any((not math.isfinite(t) or not 0 <= t <= 1 for t in result.values())):
        raise ValueError('SSIM thresholds must be finite and in [0, 1]')

    return result


def budget_bytes(throughput_bits_per_second, seconds, cap=None):
    """Decimal Mbps (1 Mbps = 1,000,000 bit/s); integer bytes, rounded down."""
    rate, duration = (Decimal(str(throughput_bits_per_second)), Decimal(str(seconds)))

    if not rate.is_finite() or not duration.is_finite() or rate < 0 or (duration < 0):
        raise ValueError('Rate and duration must be finite and nonnegative')

    value = int((rate * duration / 8).to_integral_value(rounding=ROUND_FLOOR))
    if cap is not None:
        if type(cap) is not int or cap < 0:
            raise ValueError('Byte cap must be a nonnegative integer')

        value = min(value, cap)

    return value


def validate_candidates(rows, ids):
    expected = {(i, v) for i in ids for v in ('low', 'medium', 'high')}
    actual = [(r['image_id'], r['version']) for r in rows]

    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Expected exactly three distinct candidates per configured image')

    for row in rows:
        if type(row['bytes']) is not int or row['bytes'] <= 0:
            raise ValueError('Candidate bytes must be positive integers')

        if not math.isfinite(row['ssim']) or not -1 <= row['ssim'] <= 1:
            raise ValueError('Invalid measured SSIM')

        if any((type(row[d]) is not int or row[d] < 11 for d in ('width', 'height'))):
            raise ValueError('Invalid candidate dimensions')

    for image_id in ids:
        if len({(r['width'], r['height']) for r in rows if r['image_id'] == image_id}) != 1:
            raise ValueError('Candidate dimensions must match per image')
