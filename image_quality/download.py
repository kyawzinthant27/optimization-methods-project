"""Read only selected ZIP members using verified HTTP byte ranges.

First run without --download to estimate transfer size. Never falls back to a
full ZIP download. --max-download-bytes is an explicit transfer authorization.
"""
import argparse
import io
import re
import urllib.request
import zipfile
import zlib
from pathlib import Path
from .common import config, read_json, sha256, write_json


class RangeReader(io.RawIOBase):
    def __init__(self, url, max_bytes=1000000):
        self.url, self.limit, self.transferred, self.position = (url, max_bytes, 0, 0)
        request = urllib.request.Request(
            url,
            method='HEAD',
            headers={'Accept-Encoding': 'identity'}
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            self.size = int(response.headers['Content-Length'])
            self.etag = response.headers.get('ETag')
            self.modified = response.headers.get('Last-Modified')

        if self.size <= 0:
            raise ValueError('Archive has no valid size')

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError('Negative offset')

        return self.position

    def tell(self):
        return self.position

    def read(self, length=-1):
        length = min(self.size - self.position, length if length >= 0 else self.size)
        if length <= 0:
            return b''

        if self.transferred + length > self.limit:
            raise RuntimeError('Download limit reached; inspect the plan and authorize the required bytes')

        start, end = (self.position, self.position + length - 1)
        headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}

        if self.etag and (not self.etag.startswith('W/')):
            headers['If-Match'] = self.etag
        elif self.modified:
            headers['If-Unmodified-Since'] = self.modified

        with urllib.request.urlopen(
            urllib.request.Request(self.url, headers=headers),
            timeout=60
        ) as response:
            if response.status != 206:
                raise RuntimeError('Server did not honor byte ranges; full-archive fallback is forbidden')

            content_range = response.headers.get('Content-Range', '')
            if content_range != f'bytes {start}-{end}/{self.size}':
                raise RuntimeError(f'Unexpected Content-Range: {content_range}')

            data = response.read(length + 1)

        if len(data) != length:
            raise RuntimeError('Truncated or oversized range response')

        self.transferred += len(data)
        self.position += len(data)

        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--max-download-bytes', type=int, default=1000000)
    parser.add_argument(
        '--local-zip',
        type=Path,
        help='Use an existing official archive, without network'
    )
    args = parser.parse_args()
    cfg = config(args.config)

    if not re.fullmatch(
        'https://data\\.vision\\.ee\\.ethz\\.ch/cvl/DIV2K/DIV2K_valid_HR\\.zip',
        cfg['dataset_url']
    ):
        raise ValueError('Only the official validation archive is supported')

    source = args.local_zip or RangeReader(cfg['dataset_url'], args.max_download_bytes)
    lock_path = Path('manifests/originals.json')
    existing = read_json(lock_path) if lock_path.exists() else None

    if existing and existing['dataset_url'] != cfg['dataset_url']:
        raise ValueError('Existing source lock refers to a different dataset')

    locked = {r['image_id']: r for r in existing['images']} if existing else {}

    with zipfile.ZipFile(source) as archive:
        members = [archive.getinfo(f'DIV2K_valid_HR/{i}.png') for i in cfg['image_ids']]
        transfer = sum((m.compress_size for m in members))
        plan = {
            'dataset_url': cfg['dataset_url'],
            'image_ids': cfg['image_ids'],
            'compressed_image_bytes': transfer,
            'estimated_transfer_with_metadata_bytes': transfer + 1000000,
            'uncompressed_image_bytes': sum((m.file_size for m in members)),
            'full_archive_bytes': args.local_zip.stat().st_size if args.local_zip else source.size,
            'selection_rule': 'First 14 ascending validation IDs, fixed before quality measurement' if cfg['image_ids'] == [
                f'{i:04d}'
                for i in range(801, 815)
            ] else 'Explicit ordered image IDs from configuration'
        }
        write_json('manifests/download_plan.json', plan)
        print(plan)

        if not args.download:
            return

        if not args.local_zip and transfer + 1000000 > args.max_download_bytes:
            raise RuntimeError('Transfer exceeds authorization. Review download_plan.json before raising the limit')

        records = []
        for image_id, member in zip(cfg['image_ids'], members):
            path = Path('data/originals') / f'{image_id}.png'
            path.parent.mkdir(parents=True, exist_ok=True)

            if not path.exists():
                if existing and image_id not in locked:
                    raise ValueError('Changed selection: use a separate experiment directory')

                data = archive.read(member)
                if len(data) != member.file_size:
                    raise ValueError('Incorrect extracted size')

                path.write_bytes(data)

            digest = sha256(path)
            if image_id in locked and digest != locked[image_id]['sha256']:
                raise ValueError(f'Original checksum mismatch: {image_id}')

            if path.stat().st_size != member.file_size or zlib.crc32(path.read_bytes()) != member.CRC:
                raise ValueError(f'Original disagrees with official ZIP metadata: {image_id}')

            records.append({
                'image_id': image_id,
                'path': str(path),
                'sha256': digest,
                'bytes': path.stat().st_size,
                'zip_member': member.filename,
                'zip_crc32': f'{member.CRC:08x}'
            })
            print(f'Verified {image_id}: {path.stat().st_size} bytes', flush=True)

    write_json(
        lock_path,
        {
            'dataset_url': cfg['dataset_url'],
            'source_page': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/',
            'selection_rule': plan['selection_rule'],
            'images': records,
            'archive_etag': None if args.local_zip else source.etag,
            'archive_last_modified': None if args.local_zip else source.modified,
            'downloaded_bytes_this_run': 0 if args.local_zip else source.transferred
        }
    )


if __name__ == '__main__':
    main()
