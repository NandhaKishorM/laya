"""Verify the installed wheel and repair one known NVIDIA ARM64 metadata defect.

cuSPARSELt 0.8.0 and 0.8.1 ship aarch64 wheels whose internal WHEEL file declares
an SBSA platform tag that pip rejects. PyTorch 2.11 (0.8.0) and 2.14 (0.8.1) pin them
for CUDA 13.0; NVIDIA corrected the tag in 0.9.0. Validate the actual native library
before correcting that tag. This does not change the library or relax dependency
checks for other packages.
"""
import base64
import csv
import ctypes
import hashlib
from importlib.metadata import distribution, PackageNotFoundError
import io
import platform
from pathlib import Path
import sys

# Versions whose wheels were inspected and carry only the tag defect.
REVIEWED = ("0.8.0", "0.8.1")


def repair_cusparselt(dist):
    old = "Tag: py3-none-manylinux2014_sbsa"
    wheel = next(p for p in dist.files if str(p).endswith(".dist-info/WHEEL"))
    path = Path(dist.locate_file(wheel))
    data = path.read_text(encoding="utf-8")
    if old not in data.splitlines():
        return False
    if dist.metadata["Name"] != "nvidia-cusparselt-cu13" or dist.version not in REVIEWED:
        raise RuntimeError("Unreviewed cuSPARSELt SBSA wheel; verify its version and library first")
    library = Path(dist.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0"))
    with library.open("rb") as handle:
        header = handle.read(20)
    if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183:
        raise RuntimeError("cuSPARSELt library is not ELF64 AArch64")
    ctypes.CDLL(str(library))
    updated = data.replace(old, "Tag: py3-none-manylinux2014_aarch64").encode()
    record = path.with_name("RECORD")
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    entry = next(row for row in rows if row[0] == str(wheel))
    digest = base64.urlsafe_b64encode(hashlib.sha256(updated).digest()).rstrip(b"=").decode()
    entry[1:] = ["sha256=" + digest, str(len(updated))]
    path.write_bytes(updated)
    with record.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    print(f"Corrected cuSPARSELt {dist.version} ARM64 wheel metadata")
    return True


def main():
    if platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"):
        try:
            dist = distribution("nvidia-cusparselt-cu13")
        except PackageNotFoundError:
            pass
        else:
            repair_cusparselt(dist)
    import torch

    expected = sys.argv[1]
    actual = "cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", "")
    if actual != expected:
        raise RuntimeError(f"Expected {expected} PyTorch, installed {actual}")
    if actual != "cpu":
        from torch.backends import cusparselt
        if not cusparselt.is_available():
            raise RuntimeError("CUDA build is missing cuSPARSELt")
    print(f"PyTorch {torch.__version__} imported on {platform.machine()} ({actual})")


if __name__ == "__main__":
    main()
