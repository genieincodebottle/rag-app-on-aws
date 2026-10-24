"""Build the Lambda bundle: one zip shared by the api, ingest and migrate functions.

    uv run python scripts/build_lambda.py            # -> build/ragplatform.zip

* Dependencies come from uv.lock (runtime dependencies only, no extras), so a
  build is reproducible from the commit.
* Wheels are installed for manylinux_2_28 aarch64 and Python 3.13, the Lambda
  runtime and architecture, whatever machine runs the build. Amazon Linux 2023
  has glibc 2.34, so manylinux_2_28 wheels run on it.
* The zip is deterministic: sorted entries and fixed timestamps, so the same
  inputs produce the same SHA-256.
* The base64 SHA-256 printed at the end is what Terraform's source_code_hash
  expects; the deploy workflow passes it as artifact_sha256.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
PACKAGE = BUILD / "package"
ZIP = BUILD / "ragplatform.zip"
FIXED_TIME = (2020, 1, 1, 0, 0, 0)
# Large or unused at runtime; excluding them keeps cold starts shorter.
EXCLUDE_DIRS = {"__pycache__", "tests", "test"}
EXCLUDE_SUFFIXES = {".pyc", ".pyi"}


# uv 0.9 or later understands the dual manylinux tags current wheels use.
# Override with UV="uvx --from uv>=0.9 uv" when the installed uv is older.
UV = shlex.split(os.environ.get("UV", "uv"))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    PACKAGE.mkdir(parents=True)
    requirements = BUILD / "requirements.txt"
    run(
        [
            *UV,
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--format",
            "requirements-txt",
            "--output-file",
            str(requirements),
        ]
    )
    run(
        [
            *UV,
            "pip",
            "install",
            "--target",
            str(PACKAGE),
            "--python-platform",
            "aarch64-manylinux_2_28",
            "--python-version",
            "3.13",
            "--only-binary",
            ":all:",
            "--no-deps",
            "-r",
            str(requirements),
        ]
    )
    shutil.copytree(
        ROOT / "src" / "ragplatform",
        PACKAGE / "ragplatform",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    files = sorted(
        p
        for p in PACKAGE.rglob("*")
        if p.is_file()
        and not (set(p.relative_to(PACKAGE).parts) & EXCLUDE_DIRS)
        and p.suffix not in EXCLUDE_SUFFIXES
    )
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(PACKAGE).as_posix(), date_time=FIXED_TIME)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())

    digest = hashlib.sha256(ZIP.read_bytes()).digest()
    size_mb = ZIP.stat().st_size / 1e6
    unzipped_mb = sum(p.stat().st_size for p in files) / 1e6
    print(
        f"bundle: {ZIP} ({size_mb:.1f} MB zipped, {unzipped_mb:.1f} MB unzipped, "
        f"{len(files)} files)"
    )
    if unzipped_mb > 250:
        print("ERROR: Lambda's unzipped limit is 250 MB", file=sys.stderr)
        return 1
    print(f"sha256_base64={base64.b64encode(digest).decode()}")
    print(f"sha256_hex={digest.hex()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
