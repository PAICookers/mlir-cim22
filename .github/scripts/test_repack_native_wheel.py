"""Check the platform-wheel repacker without rebuilding LLVM or CIM22."""

import hashlib
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipInfo

from wheel.wheelfile import WheelFile

spec = importlib.util.spec_from_file_location(
    "repack_native_wheel", Path(__file__).with_name("repack-native-wheel.py")
)
assert spec is not None and spec.loader is not None
repack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repack)


def _wheel(root: Path, version: str, tag: str, files: dict[str, str]) -> Path:
    metadata = f"mlir_cim22-{version}.dist-info"
    path = root / f"mlir_cim22-{version}-{tag}.whl"
    platforms = tag.removeprefix("py3-none-").split(".")
    tags = "".join(f"Tag: py3-none-{platform}\n" for platform in platforms)
    with WheelFile(path, "w") as archive:
        archive.writestr(
            f"{metadata}/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n{tags}",
        )
        archive.writestr(
            f"{metadata}/METADATA", f"Name: mlir-cim22\nVersion: {version}\n"
        )
        for name, value in files.items():
            info = ZipInfo(name.replace("METADATA_DIR", metadata))
            info.create_system = 3
            info.external_attr = 0o100755 << 16
            archive.writestr(info, value)
    return path


class NativeWheelTest(unittest.TestCase):
    def test_platform_payload_with_current_code_and_metadata(self) -> None:
        for platform, native_files in {
            "linux": {
                "cim22/bin/mlir-cim22-compile": "executable",
                "cim22/lib/libcim22_runtime.so": "library",
                "mlir_cim22.libs/libusb.so": "grafted library",
            },
            "win32": {
                "cim22/bin/mlir-cim22-compile.exe": "executable",
                "cim22/lib/cim22_runtime.dll": "library",
                "cim22/lib/cim22_runtime.lib": "import library",
            },
        }.items():
            with (
                self.subTest(platform=platform),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                tag = repack.PLATFORM_TAGS[platform]
                if platform == "linux":
                    tag += ".manylinux_2_17_x86_64.manylinux2014_x86_64"
                license_file = "licenses/third-party/VENDOR.txt"
                original = _wheel(
                    root,
                    "1.0.0",
                    tag,
                    {
                        **native_files,
                        f"METADATA_DIR/{license_file}": "vendor notice",
                        "cim22/__init__.py": "# old code",
                        "cim22/web/static/assets/old.js": "old",
                    },
                )
                prepared = _wheel(
                    root,
                    "1.0.1",
                    "py3-none-any",
                    {
                        "cim22/__init__.py": "# current code",
                        "cim22/web/static/index.html": "current page",
                        "METADATA_DIR/entry_points.txt": "current entry points",
                        "METADATA_DIR/licenses/LICENSE": "current license",
                    },
                )
                cache = root / "cache"
                repack.save(original, cache, platform)
                result = repack._repack(prepared, cache, root / "dist", platform)
                self.assertEqual(result.name, f"mlir_cim22-1.0.1-{tag}.whl")
                checksum = hashlib.sha256(result.read_bytes()).hexdigest()
                self.assertEqual(
                    result.with_suffix(".whl.sha256").read_text(),
                    f"{checksum}  {result.name}\n",
                )
                metadata = "mlir_cim22-1.0.1.dist-info"
                with WheelFile(result) as wheel:
                    # WheelFile checks each entry against the regenerated RECORD.
                    self.assertIsNone(wheel.testzip())
                    for name, value in native_files.items():
                        self.assertEqual(wheel.read(name), value.encode())
                        self.assertEqual(
                            wheel.getinfo(name).external_attr >> 16, 0o100755
                        )
                    self.assertEqual(
                        wheel.read(f"{metadata}/{license_file}"), b"vendor notice"
                    )
                    self.assertIn(b"Version: 1.0.1", wheel.read(f"{metadata}/METADATA"))
                    header = wheel.read(f"{metadata}/WHEEL")
                    for alias in tag.removeprefix("py3-none-").split("."):
                        self.assertIn(f"Tag: py3-none-{alias}\n".encode(), header)
                    self.assertEqual(
                        wheel.read(f"{metadata}/entry_points.txt"),
                        b"current entry points",
                    )
                    self.assertEqual(
                        wheel.read(f"{metadata}/licenses/LICENSE"),
                        b"current license",
                    )
                    self.assertEqual(
                        wheel.read("cim22/web/static/index.html"), b"current page"
                    )
                    self.assertEqual(wheel.read("cim22/__init__.py"), b"# current code")
                    self.assertNotIn("cim22/web/static/assets/old.js", wheel.namelist())
                incompatible = "win32" if platform == "linux" else "linux"
                with self.assertRaisesRegex(ValueError, "tagged"):
                    repack._repack(prepared, cache, root / "wrong", incompatible)

    def test_reject_duplicate_native_path(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = _wheel(
                root,
                "1.0.0",
                repack.PLATFORM_TAGS["linux"],
                {
                    "cim22/bin/tool": "native executable",
                    "cim22/lib/runtime": "native library",
                    "METADATA_DIR/licenses/third-party/VENDOR.txt": "notice",
                },
            )
            prepared = _wheel(
                root, "1.0.1", "py3-none-any", {"cim22/bin/tool": "duplicate"}
            )
            cache = root / "cache"
            repack.save(original, cache, "linux")
            with self.assertRaisesRegex(ValueError, "overlapping native entry"):
                repack._repack(prepared, cache, root / "dist", "linux")
            self.assertEqual(list((root / "dist").glob("*.whl")), [])


if __name__ == "__main__":
    unittest.main()
