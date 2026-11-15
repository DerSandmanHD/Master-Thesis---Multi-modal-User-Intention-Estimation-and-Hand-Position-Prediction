"""Release validation must reject corruption and unsafe paths without ML imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'reproducibility/artifacts.py'
spec = importlib.util.spec_from_file_location('release_artifacts', MODULE)
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


class ArtifactTests(unittest.TestCase):
    def test_hash_detects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'example.csv'
            p.write_bytes(b'abc')
            entry = {'path': p.name, 'size_bytes': 3, 'sha256': artifacts.digest(p)}
            self.assertEqual(artifacts.verify(tmp, [entry])[0][1], 'verified')
            p.write_bytes(b'xyz')
            self.assertEqual(artifacts.verify(tmp, [entry])[0][1], 'sha256_mismatch')
            p.unlink()
            self.assertEqual(artifacts.verify(tmp, [entry])[0][1], 'missing')

    def test_release_rejects_traversal_duplicates_and_raw_media(self):
        for paths in [['../escape'], ['/absolute'], ['a', 'a'], ['recording.vrs']]:
            with self.subTest(paths=paths), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / 'lock.json'
                p.write_text(json.dumps({'schema_version': 1, 'files': [
                    {'path': x, 'sha256': '0' * 64, 'groups': ['data']} for x in paths]}))
                with self.assertRaises(ValueError):
                    artifacts.load_lock(p)

    def test_unknown_profile_and_duplicate_group_selection(self):
        lock = {'files': [{'path': 'a', 'groups': ['data', 'main']}]}
        self.assertEqual(len(artifacts.selected(lock, ['data', 'main'])), 1)
        with self.assertRaises(ValueError):
            artifacts.selected(lock, ['unknown'])

    def test_symlink_outside_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            target = Path(other) / 'external'
            target.write_bytes(b'abc')
            (Path(root) / 'link').symlink_to(target)
            entry = {'path': 'link', 'size_bytes': 3, 'sha256': artifacts.digest(target)}
            self.assertEqual(artifacts.verify(root, [entry])[0][1], 'outside_root')
