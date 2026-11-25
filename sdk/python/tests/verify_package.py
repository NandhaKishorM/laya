"""Verify built distributions in a client environment with no inference dependencies."""
import argparse
from email.parser import BytesParser
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile


def command(*args, cwd, env):
    subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True)


def check_wheel(path):
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
        assert len(metadata) == 1
        message = BytesParser().parsebytes(wheel.read(metadata[0]))
        assert message["Name"] == "laya-client"
        assert not message.get_all("Requires-Dist"), message.get_all("Requires-Dist")
        assert "laya_client/py.typed" in names
        assert any("LICENSE" in n for n in names)
        assert all(n.startswith(("laya_client/", "laya_client-")) for n in names), names


def extract_sdist(source, destination):
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Package verification requires a patched Python with tarfile.data_filter")
    destination = destination.resolve()
    with tarfile.open(source) as archive:
        for entry in archive.getmembers():
            target = (destination / entry.name).resolve()
            if Path(entry.name).is_absolute() or not target.is_relative_to(destination):
                raise ValueError("sdist member escapes the extraction directory")
            if not (entry.isfile() or entry.isdir()):
                raise ValueError("sdist must contain only regular files and directories")
        archive.extractall(destination, filter="data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument("--server-python", type=Path)
    parser.add_argument("--legacy-server-fixture", type=Path)
    args = parser.parse_args()
    tests = Path(__file__).resolve().parent
    dist = args.dist.resolve()
    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "Use a dist directory containing exactly one wheel and sdist"
    check_wheel(wheels[0])
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    with tempfile.TemporaryDirectory(prefix="laya-sdk-package-") as directory:
        root = Path(directory)
        environment = root / "client"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command(python, "-m", "pip", "install", "--no-index", wheels[0], cwd=root, env=env)
        command(python, "-I", "-m", "unittest", "discover", "-s", tests, "-p", "test_*.py",
                "-v", cwd=root, env=env)
        print("PASS wheel installation with normal resolution and no package index", flush=True)

        unpacked = root / "sdist"
        unpacked.mkdir()
        extract_sdist(sdists[0], unpacked)
        projects = list(unpacked.iterdir())
        assert len(projects) == 1 and (projects[0] / "pyproject.toml").is_file()
        rebuilt = root / "rebuilt"
        command(sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", rebuilt,
                projects[0], cwd=root, env=env)
        rebuilt_wheels = list(rebuilt.glob("*.whl"))
        assert len(rebuilt_wheels) == 1
        check_wheel(rebuilt_wheels[0])
        command(python, "-m", "pip", "install", "--no-index", "--force-reinstall", rebuilt_wheels[0],
                cwd=root, env=env)
        command(python, "-I", "-m", "unittest", "discover", "-s", tests, "-p", "test_*.py",
                cwd=root, env=env)
        print("PASS sdist rebuild and installed-package regression", flush=True)
        if args.server_python:
            command(python, "-I", tests / "integration.py", "--server-python", args.server_python.resolve(),
                    cwd=root, env=env)
            if args.legacy_server_fixture:
                command(python, "-I", tests / "integration.py", "--server-python", args.server_python.resolve(),
                        "--server-fixture", args.legacy_server_fixture.resolve(), "--legacy-fixture",
                        cwd=root, env=env)
        elif args.legacy_server_fixture:
            parser.error("--legacy-server-fixture requires --server-python")


if __name__ == "__main__":
    main()
