"""Set the CI release or development version before building the sdist."""

import os
import re
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

RELEASE_VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"


def wheel_version(
    base: str, ref_type: str, ref_name: str, date: str, run_number: str
) -> str:
    """Require matching release tags; derive branch snapshots from the target."""
    if not re.fullmatch(RELEASE_VERSION, base):
        raise ValueError("Expected project version X.Y.Z")
    if ref_type == "tag":
        if ref_name != f"v{base}":
            raise ValueError(f"Expected release tag v{base}, got {ref_name!r}")
        return base
    if ref_type != "branch":
        raise ValueError(f"Unsupported Git ref type: {ref_type!r}")
    if not re.fullmatch(r"[0-9]{8}", date):
        raise ValueError("Expected commit date YYYYMMDD")
    datetime.strptime(date, "%Y%m%d")
    if not re.fullmatch(r"[1-9][0-9]{0,5}", run_number):
        raise ValueError("Expected workflow run number 1..999999")
    return f"{base}.dev{date}{int(run_number):06d}"


def prepare_wheel(project: Path, version: str) -> None:
    """Write one validated version into the build checkout only."""
    if not re.fullmatch(rf"{RELEASE_VERSION}(?:\.dev[0-9]+)?", version):
        raise ValueError("Expected release X.Y.Z or development X.Y.Z.devN")
    path = project / "pyproject.toml"
    source = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'^version = "[^"]+"$',
        f'version = "{version}"',
        source,
        flags=re.MULTILINE,
    )
    if count != 1 or tomllib.loads(updated)["project"]["version"] != version:
        raise ValueError("Expected exactly one static project version")
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    """Use the same Git ref, commit date and workflow number on both runners."""
    root = Path(__file__).resolve().parents[2]
    project = root / "python"
    with (project / "pyproject.toml").open("rb") as metadata:
        base = tomllib.load(metadata)["project"]["version"]
    timestamp = subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", "HEAD"], cwd=root, text=True
    )
    date = datetime.fromtimestamp(int(timestamp), UTC).strftime("%Y%m%d")
    version = wheel_version(
        base,
        os.environ["GITHUB_REF_TYPE"],
        os.environ["GITHUB_REF_NAME"],
        date,
        os.environ["GITHUB_RUN_NUMBER"],
    )
    prepare_wheel(project, version)
    print(f"Wheel version: {version}")


if __name__ == "__main__":
    main()
