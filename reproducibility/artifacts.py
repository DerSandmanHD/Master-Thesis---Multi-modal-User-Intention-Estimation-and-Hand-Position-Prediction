#!/usr/bin/env python3
"""Inspect the processed-data release without importing ML packages or writing files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

DEFAULT_LOCK = Path(__file__).with_name('processed_artifacts.lock.json')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_lock(path=DEFAULT_LOCK):
    lock = json.loads(Path(path).read_text())
    if lock.get('schema_version') != 1 or not lock.get('files'):
        raise ValueError('Unsupported or empty artifact lock')
    seen = set()
    for item in lock['files']:
        p = Path(item['path'])
        if p.is_absolute() or '..' in p.parts or str(p) in seen:
            raise ValueError('Unsafe or duplicate artifact path')
        if p.suffix.lower() in {'.vrs', '.mp4', '.mov', '.wav'}:
            raise ValueError('Raw recordings are outside this release')
        if len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256']):
            raise ValueError('Invalid SHA-256')
        seen.add(str(p))
    return lock


def selected(lock, groups):
    known = {g for item in lock['files'] for g in item['groups']}
    if set(groups) - known:
        raise ValueError('Unknown groups: ' + ', '.join(sorted(set(groups) - known)))
    return [item for item in lock['files'] if not groups or set(groups) & set(item['groups'])]


def verify(root, items):
    """Return separate missing, unreadable, size and hash failure statuses."""
    results = []
    root = Path(root).resolve()
    for item in items:
        path = root / item['path']
        try:
            # A release must not use a symlink to read outside the selected root.
            if not path.resolve().is_relative_to(root):
                status = 'outside_root'
            elif not path.is_file():
                status = 'missing'
            elif path.stat().st_size != item['size_bytes']:
                status = 'size_mismatch'
            elif digest(path) != item['sha256']:
                status = 'sha256_mismatch'
            else:
                status = 'verified'
        except PermissionError:
            status = 'permission_denied'
        except OSError:
            status = 'io_error'
        results.append((item, status))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['summary', 'verify', 'files'])
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--groups', nargs='+', default=[])
    args = parser.parse_args()
    lock = load_lock(args.lock)
    items = selected(lock, args.groups)
    if args.command == 'files':
        for item in items:
            print(item['path'])
        return 0
    if args.command == 'summary':
        print(json.dumps({'files': len(items), 'bytes': sum(x['size_bytes'] for x in items),
                          'groups': dict(Counter(g for x in items for g in x['groups'])),
                          'scope': lock['scope']}, indent=2))
        return 0
    results = verify(args.root, items)
    failures = [{'entry': i, 'status': status} for i, (_, status) in enumerate(results)
                if status != 'verified']
    print(json.dumps({'counts': dict(Counter(status for _, status in results)),
                      'failures': failures, 'index': 'zero-based position in selected lock entries'}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
