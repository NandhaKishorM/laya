import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from verify_package import extract_sdist


@unittest.skipUnless(hasattr(tarfile, "data_filter"), "Requires patched tarfile")
class ArchiveExtractionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.destination = self.root / "unpacked"
        self.destination.mkdir()
        self.source = self.root / "source.tar.gz"

    def write_archive(self, *members):
        with tarfile.open(self.source, "w:gz") as archive:
            for name, kind, link in members:
                entry = tarfile.TarInfo(name)
                entry.type = kind
                entry.linkname = link
                if entry.isfile():
                    entry.size = len(b"package contents")
                    archive.addfile(entry, io.BytesIO(b"package contents"))
                else:
                    archive.addfile(entry)

    def test_regular_files_and_directories(self):
        self.write_archive(("package", tarfile.DIRTYPE, ""),
                           ("package/README.md", tarfile.REGTYPE, ""))
        extract_sdist(self.source, self.destination)
        self.assertEqual((self.destination / "package/README.md").read_bytes(), b"package contents")

    def test_parent_traversal_is_rejected_before_extracting_any_member(self):
        self.write_archive(("safe.txt", tarfile.REGTYPE, ""), ("../outside.txt", tarfile.REGTYPE, ""))
        with self.assertRaisesRegex(ValueError, "escapes"):
            extract_sdist(self.source, self.destination)
        self.assertFalse((self.root / "outside.txt").exists())
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_absolute_path_is_rejected(self):
        outside = self.root / "outside.txt"
        self.write_archive((outside.as_posix(), tarfile.REGTYPE, ""))
        with self.assertRaisesRegex(ValueError, "escapes"):
            extract_sdist(self.source, self.destination)
        self.assertFalse(outside.exists())

    def test_links_are_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with self.subTest(kind=kind):
                self.write_archive(("link", kind, "../outside.txt"))
                with self.assertRaisesRegex(ValueError, "only regular files and directories"):
                    extract_sdist(self.source, self.destination)
                self.assertEqual(list(self.destination.iterdir()), [])

    def test_special_files_are_rejected(self):
        for kind in (tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
            with self.subTest(kind=kind):
                self.write_archive(("special", kind, ""))
                with self.assertRaisesRegex(ValueError, "only regular files and directories"):
                    extract_sdist(self.source, self.destination)
                self.assertEqual(list(self.destination.iterdir()), [])

    def test_existing_directory_symlink_cannot_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        try:
            (self.destination / "link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Directory symlinks unavailable: {error}")
        self.write_archive(("link/file.txt", tarfile.REGTYPE, ""))
        with self.assertRaisesRegex(ValueError, "escapes"):
            extract_sdist(self.source, self.destination)
        self.assertEqual(list(outside.iterdir()), [])


class ExtractionFilterTests(unittest.TestCase):
    def test_missing_filter_fails_before_opening_the_archive(self):
        with patch.dict(vars(tarfile)):
            vars(tarfile).pop("data_filter", None)
            with patch.object(tarfile, "open") as open_archive:
                with self.assertRaisesRegex(RuntimeError, "patched Python"):
                    extract_sdist(Path("unused.tar.gz"), Path("unused"))
                open_archive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
