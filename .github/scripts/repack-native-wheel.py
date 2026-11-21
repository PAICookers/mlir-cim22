"""Reuse a validated platform wheel's native payload with current Python and Web."""

import argparse
import copy
import hashlib
import re
import subprocess
import sys
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from wheel.wheelfile import WheelFile

NATIVE_PREFIXES = ("cim22/bin/", "cim22/lib/", "cim22/share/", "mlir_cim22.libs/")
THIRD_PARTY = "licenses/third-party/"
CACHE_NAME = "native.zip"
WHEEL_HEADER = "WHEEL"
PLATFORM_TAGS = {
    "linux": "py3-none-manylinux_2_28_x86_64",
    "win32": "py3-none-win_amd64",
}


def _entries(archive: ZipFile) -> dict[str, ZipInfo]:
    entries = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if (
            name in entries
            or "\\" in name
            or PurePosixPath(name).as_posix() != name
            or ".." in name.split("/")
            or ":" in name
            or name.startswith("/")
        ):
            raise ValueError(f"Invalid or duplicate archive entry: {name}")
        entries[name] = info
    return entries


def _metadata_dir(entries: dict[str, ZipInfo]) -> str:
    (path,) = [
        PurePosixPath(name) for name in entries if name.endswith(".dist-info/WHEEL")
    ]
    if len(path.parts) != 2:
        raise ValueError(f"Expected top-level wheel metadata: {path}")
    return path.parent.name


def _native_path(name: str, metadata: str) -> str | None:
    if name.startswith(NATIVE_PREFIXES):
        return name
    if name.startswith(f"{metadata}/{THIRD_PARTY}"):
        return name.removeprefix(f"{metadata}/")
    return None


def _copy_entry(source: ZipFile, entry: ZipInfo, dest: ZipFile, name: str) -> None:
    # Preserve Unix executable bits for the cached compiler tools.
    info = copy.copy(entry)
    info.filename = name
    dest.writestr(info, source.read(entry))


def _platform_tags(header: bytes, platform: str) -> list[str]:
    required = PLATFORM_TAGS[platform]
    fields = BytesParser().parsebytes(header)
    tags = fields.get_all("Tag", [])
    compatible = (
        all(
            re.fullmatch(
                r"py3-none-manylinux(?:_[0-9]+_[0-9]+|1|2010|2014)_x86_64",
                tag,
            )
            for tag in tags
        )
        if platform == "linux"
        else tags == [required]
    )
    if (
        required not in tags
        or not compatible
        or len(set(tags)) != len(tags)
        or fields.get("Root-Is-Purelib") != "false"
    ):
        raise ValueError(f"Expected a platform wheel tagged {required}")
    return tags


def save(wheel: Path, cache_dir: Path, platform: str) -> None:
    """Keep just the native files, platform tags and third-party notices."""
    with WheelFile(wheel) as archive:
        entries = _entries(archive)
        metadata = _metadata_dir(entries)
        header = archive.read(f"{metadata}/{WHEEL_HEADER}")
        _platform_tags(header, platform)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache_dir) as directory:
            cached = Path(directory) / CACHE_NAME
            with ZipFile(cached, "w", compression=ZIP_DEFLATED) as dest:
                dest.writestr(WHEEL_HEADER, header)
                for name, info in entries.items():
                    if native := _native_path(name, metadata):
                        _copy_entry(archive, info, dest, native)
            cached.replace(cache_dir / CACHE_NAME)
    print(f"Saved native payload from {wheel.name}")


def _repack(source_wheel: Path, cache_dir: Path, out_dir: Path, platform: str) -> Path:
    with (
        WheelFile(source_wheel) as source,
        ZipFile(cache_dir / CACHE_NAME) as cache,
    ):
        source_entries = _entries(source)
        native_entries = _entries(cache)
        metadata = _metadata_dir(source_entries)
        wheel_header = f"{metadata}/{WHEEL_HEADER}"
        tags = _platform_tags(cache.read(WHEEL_HEADER), platform)

        native_entries.pop(WHEEL_HEADER)
        if not any(name.startswith("cim22/bin/") for name in native_entries):
            raise ValueError("Cached native payload has no compiler tools")
        if not any(name.startswith("cim22/lib/") for name in native_entries):
            raise ValueError("Cached native payload has no runtime libraries")
        if not any(name.startswith(THIRD_PARTY) for name in native_entries):
            raise ValueError("Cached native payload has no third-party notices")

        fields = BytesParser().parsebytes(source.read(wheel_header))
        del fields["Tag"]
        for tag in tags:
            fields["Tag"] = tag
        fields.replace_header("Root-Is-Purelib", "false")
        platforms = ".".join(tag.removeprefix("py3-none-") for tag in tags)
        filename = f"{metadata.removesuffix('.dist-info')}-py3-none-{platforms}.whl"

        out_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=out_dir) as directory:
            staged = Path(directory) / filename
            with WheelFile(staged, "w") as wheel:
                for name, info in source_entries.items():
                    if name not in {wheel_header, f"{metadata}/RECORD"}:
                        _copy_entry(source, info, wheel, name)
                for name, info in native_entries.items():
                    if name.startswith(THIRD_PARTY):
                        name = f"{metadata}/{name}"
                    if name in source_entries or _native_path(name, metadata) is None:
                        raise ValueError(f"Invalid or overlapping native entry: {name}")
                    _copy_entry(cache, info, wheel, name)
                wheel.writestr(wheel_header, fields.as_bytes())
            target = staged.replace(out_dir / filename)

    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    target.with_suffix(".whl.sha256").write_text(
        f"{digest}  {target.name}\n", encoding="ascii", newline="\n"
    )
    return target


def build(cache_dir: Path, out_dir: Path, platform: str) -> None:
    """Build the current non-native files without configuring CMake."""
    project = Path(__file__).resolve().parents[2] / "python"
    with TemporaryDirectory() as directory:
        subprocess.run(
            [
                "uv",
                "build",
                "--python",
                "3.12",
                str(project),
                "--wheel",
                "--out-dir",
                directory,
                "-Cwheel.cmake=false",
                "-Cwheel.platlib=true",
            ],
            check=True,
        )
        (source_wheel,) = Path(directory).glob("*.whl")
        target = _repack(source_wheel, cache_dir, out_dir, platform)
    print(f"Repacked {target.name} with cached native payload")


def main() -> None:
    """Save a complete wheel's payload or assemble a wheel from the cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    save_parser = commands.add_parser("save")
    save_parser.add_argument("wheel", type=Path)
    save_parser.add_argument("cache_dir", type=Path)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("cache_dir", type=Path)
    build_parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    if args.command == "save":
        save(args.wheel, args.cache_dir, sys.platform)
    else:
        build(args.cache_dir, args.out_dir, sys.platform)


if __name__ == "__main__":
    main()
