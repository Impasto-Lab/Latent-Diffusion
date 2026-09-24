"""Verify that truncated or corrupted model downloads cannot pass integrity checks."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.download_model import verify_file


class DownloadTests(unittest.TestCase):
    def test_lfs_rejects_same_length_corruption_and_truncation(self):
        content = b"test model weights"
        entry = {"size": len(content), "lfs": {"sha256": hashlib.sha256(content).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(content)
            self.assertTrue(verify_file(path, entry))
            path.write_bytes(b"X" + content[1:])
            self.assertFalse(verify_file(path, entry))
            path.write_bytes(content[:-1])
            self.assertFalse(verify_file(path, entry))

    def test_git_blob_digest_for_configuration_file(self):
        content = b"{}\n"
        entry = {"size": len(content), "blobId": hashlib.sha1(b"blob 3\0" + content).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(content)
            self.assertTrue(verify_file(path, entry))


if __name__ == "__main__":
    unittest.main()
