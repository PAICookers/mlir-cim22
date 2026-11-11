"""Check native files required by the installed CIM22 wheel."""

import sys
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check-wheel-contents.py WHEEL")
    wheel = Path(sys.argv[1])
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
    windows = "win_amd64" in wheel.name
    required = {
        "cim22/bin/mlir-cim22-opt.exe" if windows else "cim22/bin/mlir-cim22-opt",
        "cim22/bin/mlir-cim22-compile.exe"
        if windows
        else "cim22/bin/mlir-cim22-compile",
        "cim22/lib/cim22_runtime.dll" if windows else "cim22/lib/libcim22_runtime.so",
    }
    required.add("cim22/lib/cim22_runtime.lib" if windows else "cim22/lib/libcyusb.so")
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"Wheel is missing: {', '.join(missing)}")
    forbidden = {
        name
        for name in names
        if name.endswith(
            ("cim22_compiler.dll", "cim22_compiler.lib", "libcim22_compiler.so")
        )
    }
    if forbidden:
        raise SystemExit(
            f"Wheel contains retired compiler libraries: {sorted(forbidden)}"
        )
    print(f"Wheel contents: {wheel.name}: PASS")


if __name__ == "__main__":
    main()
