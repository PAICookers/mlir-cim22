"""Durable task files and fixed DVS input identity.

CSV is the commit record: a completed pair is replaced atomically before the
status is updated. Recovery derives progress and totals from that CSV.
"""

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, TypedDict

# Match the verified DatasetFolder lexicographic class mapping, including '10'.
CLASSES = sorted(map(str, range(11)))
INPUT_SHAPE = [32, 1, 2, 128, 128]
ACTIVE = {"starting", "validating", "loading", "compiling", "running"}
type Document = dict[str, Any]  # Persisted JSON documents, validated on use.


class Result(TypedDict):
    index: int
    sample: str
    label: int
    original_prediction: int
    cim_prediction: int
    original_correct: bool
    cim_correct: bool
    original_seconds: float
    cim_seconds: float


FIELDS = list(Result.__annotations__)


def digest(path: Path) -> str:
    """Hash file contents for restart identity checks."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path: Path, content: str) -> None:
    """Publish a durable whole file, never a partially written CSV/JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Document:
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(directory: Path, rows: list[Result]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(directory / "results.csv", output.getvalue())


def read_rows(directory: Path) -> list[Result]:
    path = directory / "results.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("结果 CSV 表头无效，无法续跑")
        raw = list(reader)
    rows: list[Result] = []
    for i, row in enumerate(raw):
        if any(value is None for value in row.values()) or None in row:
            raise ValueError("结果 CSV 存在不完整行，无法续跑")
        for side in ("original", "cim"):
            if row[f"{side}_correct"] not in {"True", "False"}:
                raise ValueError("结果 CSV 正误字段无效")
            seconds = float(row[f"{side}_seconds"])
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("结果 CSV 耗时无效")
            label = int(row["label"])
            prediction = int(row[f"{side}_prediction"])
            if not 0 <= label < 11 or not 0 <= prediction < 11:
                raise ValueError("结果 CSV 类别索引无效")
            if (prediction == label) != (row[f"{side}_correct"] == "True"):
                raise ValueError("结果 CSV 预测与正误不一致")
        parsed = {
            key: (row[key] == "True")
            if key.endswith("_correct")
            else float(row[key])
            if key.endswith("_seconds")
            else row[key]
            if key == "sample"
            else int(row[key])
            for key in FIELDS
        }
        if parsed["index"] != i:
            raise ValueError("结果 CSV 不是连续样本前缀，无法续跑")
        rows.append(Result(**parsed))
    return rows


def totals(rows: list[Result]) -> Document:
    count = len(rows)
    return {
        side: {
            "correct": sum(row[f"{side}_correct"] for row in rows),
            "accuracy": sum(row[f"{side}_correct"] for row in rows) / count
            if count
            else None,
            "seconds": sum(row[f"{side}_seconds"] for row in rows),
        }
        for side in ("original", "cim")
    }


def make_config(model_dir: Path, weights: Path, dataset: Path) -> Document:
    """Index cached frames using the fixed training label order."""
    source = model_dir / "model_mobilenet_snn_deploy.py"
    for path in (source, weights):
        if not path.is_file():
            raise ValueError(f"找不到文件：{path}")
    if not dataset.is_dir():
        raise ValueError("测试集目录不存在")
    unknown = {p.name for p in dataset.iterdir() if p.is_dir()} - set(CLASSES)
    if unknown:
        raise ValueError(f"未知类别目录：{sorted(unknown)}")
    entries = [
        {
            "path": path.relative_to(dataset).as_posix(),
            "label": label,
            "sha256": digest(path),
        }
        for label, name in enumerate(CLASSES)
        for path in sorted((dataset / name).glob("*.npz"))
    ]
    if not entries:
        raise ValueError("目录中没有 <类别 0–10>/*.npz 缓存样本")
    return {
        "model_dir": str(model_dir),
        "weights": str(weights),
        "dataset": str(dataset),
        "source_sha256": digest(source),
        "weights_sha256": digest(weights),
        "samples": entries,
        "classes": CLASSES,
        "input_shape": INPUT_SHAPE,
        "runtime": "software",
    }


def validate_config(config: Document) -> None:
    """Refuse to mix changed inputs with previous sample results."""
    current = make_config(
        Path(config["model_dir"]),
        Path(config["weights"]),
        Path(config["dataset"]),
    )
    if current != config:
        raise ValueError("模型、权重或测试集已变化，请重新导入为新任务")
