"""Check the bounded legal and packaging surface of a LatticeForge release."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]

REQUIRED = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "PROVENANCE.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "LatticeForge/eigen/COPYING.README",
    "LatticeForge/eigen/COPYING.MPL2",
    "LatticeForge/eigen/COPYING.BSD",
    "LatticeForge/eigen/COPYING.MINPACK",
    "LatticeForge/eigen/COPYING.APACHE",
    "LatticeForge/eigen/COPYING.LGPL",
)

FORBIDDEN_TRACKED_SUFFIXES = (".so", ".dylib", ".dll", ".whl", ".pyc")


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required release file: {relative}")

    tracked = tracked_files()
    for relative in tracked:
        if relative.endswith(FORBIDDEN_TRACKED_SUFFIXES):
            errors.append(f"tracked build artifact: {relative}")

    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    for expected in ("license='MIT'",):
        if expected not in setup_text:
            errors.append(f"setup.py is missing release metadata: {expected}")

    for relative, expected in (
        ("LICENSE", "MIT License"),
        ("CITATION.cff", "license: MIT"),
        ("README.md", "MIT licence in [`LICENSE`](LICENSE)"),
        ("THIRD_PARTY_NOTICES.md", "The MIT"),
    ):
        if expected not in (ROOT / relative).read_text(encoding="utf-8"):
            errors.append(f"{relative} is missing licence declaration: {expected}")

    setup_version = re.search(r"version='([^']+)'", setup_text)
    citation_version = re.search(
        r"^version:\s*(\S+)", (ROOT / "CITATION.cff").read_text(encoding="utf-8"), re.M
    )
    changelog_text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if not setup_version or not citation_version:
        errors.append("could not read the version from setup.py or CITATION.cff")
    else:
        if setup_version.group(1) != citation_version.group(1):
            errors.append(
                f"version mismatch: setup.py {setup_version.group(1)}, "
                f"CITATION.cff {citation_version.group(1)}"
            )
        if f"## {setup_version.group(1)}" not in changelog_text:
            errors.append(f"CHANGELOG.md has no entry for {setup_version.group(1)}")

    manifest_text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for expected in (
        "THIRD_PARTY_NOTICES.md",
        "recursive-include LatticeForge/eigen COPYING.*",
    ):
        if expected not in manifest_text:
            errors.append(f"MANIFEST.in does not include: {expected}")

    if errors:
        print("release check: REFUSED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("release check: PASS")
    print(f"tracked files checked: {len(tracked)}")
    print(f"required legal/provenance files: {len(REQUIRED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
