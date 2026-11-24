"""Create canonical RGB references once, then encode and measure 42 candidates."""
import argparse
import io
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms, ImageOps, features
from skimage.metrics import structural_similarity
from .common import config, read_json, sha256, validate_candidates, write_csv, write_json

SSIM_SETTINGS = dict(
    channel_axis=-1,
    data_range=255,
    win_size=11,
    gaussian_weights=True,
    sigma=1.5,
    use_sample_covariance=False,
    K1=0.01,
    K2=0.03
)


def measure(reference, candidate):
    a, b = (np.asarray(reference), np.asarray(candidate))
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('SSIM requires equal-sized RGB images')

    if a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError('SSIM expects 8-bit channels with data_range=255')

    return float(structural_similarity(a, b, **SSIM_SETTINGS))


def canonical_image(path, max_edge):
    with Image.open(path) as original:
        source_size = list(original.size)
        image = ImageOps.exif_transpose(original)

        if image.mode != 'RGB':
            raise ValueError(f'Expected opaque RGB source, got {image.mode}: {path}')

        icc = image.info.get('icc_profile')
        if icc:
            image = ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                ImageCms.createProfile('sRGB'),
                renderingIntent=0,
                outputMode='RGB'
            )

        color_handling = 'ICC to sRGB, perceptual intent' if icc else 'Untagged RGB assumed sRGB'
        scale = min(1.0, max_edge / max(image.size))
        size = tuple((max(1, round(d * scale)) for d in image.size))

        if min(size) < 11:
            raise ValueError('Reference is too small for the SSIM window')

        reference = image.resize(size, Image.Resampling.LANCZOS, reducing_gap=None)
        reference.info.clear()

        return (reference, source_size, color_handling)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()
    cfg = config(args.config)
    originals = read_json('manifests/originals.json')

    if originals['dataset_url'] != cfg['dataset_url']:
        raise ValueError('Original manifest source does not match configuration')

    if [r['image_id'] for r in originals['images']] != cfg['image_ids']:
        raise ValueError('Original manifest does not match configured batch')

    candidates, references = ([], [])

    for original in originals['images']:
        image_id, path = (original['image_id'], Path(original['path']))
        if sha256(path) != original['sha256']:
            raise ValueError(f'Original checksum mismatch: {path}')

        reference, source_size, color = canonical_image(path, cfg['max_edge'])
        ref_path = Path('data/references') / f'{image_id}.png'
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        reference.save(ref_path, format='PNG', optimize=False, compress_level=9)
        references.append({
            'image_id': image_id,
            'path': str(ref_path),
            'sha256': sha256(ref_path),
            'width': reference.width,
            'height': reference.height,
            'original_dimensions': source_size,
            'color_handling': color
        })

        for label, quality in cfg['qualities'].items():
            out = Path('data/candidates') / f'{image_id}-{label}.webp'
            out.parent.mkdir(parents=True, exist_ok=True)
            reference.save(
                out,
                format='WEBP',
                quality=quality,
                **cfg['encoder'],
                icc_profile=b'',
                exif=b'',
                xmp=b''
            )

            with Image.open(out) as decoded:
                score = measure(reference, decoded.convert('RGB'))
                if decoded.size != reference.size:
                    raise ValueError('Encoder changed dimensions')

            candidates.append({
                'image_id': image_id,
                'version': label,
                'path': str(out),
                'width': reference.width,
                'height': reference.height,
                'bytes': out.stat().st_size,
                'ssim': score,
                'quality': quality,
                'sha256': sha256(out),
                'reference_path': str(ref_path)
            })

        print(f'Encoded and measured {image_id}', flush=True)

    validate_candidates(candidates, cfg['image_ids'])
    write_json('manifests/references.json', references)
    write_json('manifests/candidates.json', candidates)
    write_csv('manifests/candidates.csv', candidates, list(candidates[0]))
    write_json(
        'manifests/generation.json',
        {
            'config': cfg,
            'config_sha256': sha256(args.config),
            'ssim_settings': SSIM_SETTINGS,
            'python': platform.python_version(),
            'packages': {p: version(p) for p in ('Pillow', 'numpy', 'scipy', 'scikit-image')},
            'libwebp': features.version('webp'),
            'littlecms': features.version('littlecms2'),
            'resize': 'Single LANCZOS resize from oriented, color-normalized original; max edge; no upscaling; nearest integer dimensions',
            'color': 'Gamma-encoded sRGB RGB uint8; mean of three channel SSIMs; no luminance conversion',
            'webp': 'Lossy libwebp YUV 4:2:0; encoder settings fixed; no metadata'
        }
    )


if __name__ == '__main__':
    main()
