"""Regenerate the BF16 dot-product error figure with Matplotlib and Noto CJK."""

import csv
from pathlib import Path

from matplotlib import rc_context
from matplotlib.figure import Figure

_SERIES = (
    ("signed-exact", "共享尺度量化\n完整整数和", "D", "#3e546c"),
    ("split-exact", "按符号分组量化\n完整整数和", "s", "#996122"),
    ("sign-split", "当前补偿（模拟器）\nBF16 截断回读与主机恢复", "o", "#007468"),
)
_FIGURES = Path(__file__).resolve().parents[1]


def main() -> None:
    """Read the plotted percentiles and write the companion PNG."""
    with (_FIGURES / "data" / "bf16-dot-error.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    index = {row["case"]: i for i, row in enumerate(rows)}

    with rc_context({"font.family": "Noto Sans CJK SC", "font.size": 10}):
        figure = Figure(figsize=(7.6, 3.7))
        axis = figure.subplots()
        figure.subplots_adjust(left=0.23, right=0.96, bottom=0.16, top=0.79)

        for offset, (mode, label, marker, color) in enumerate(_SERIES, start=-1):
            axis.scatter(
                [100 * float(row[mode]) for row in rows],
                [i + 0.24 * offset for i in range(len(rows))],
                marker=marker,
                color=color,
                s=28,
                label=label,
                zorder=3,
            )

        axis.set_xscale("log")
        axis.set_xlim(0.14, 63)
        ticks = (0.2, 0.5, 1, 2, 5, 10, 20, 50)
        axis.set_xticks(ticks, [f"{value:g}%" for value in ticks])
        axis.set_yticks(range(len(rows)), [row["label"] for row in rows])
        axis.set_ylim(len(rows) - 0.5, -0.5)
        axis.set_xlabel("归一化绝对误差 P99（对数刻度）")
        axis.set_ylabel("点积用例", labelpad=8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#777777")
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", which="minor", length=0)
        axis.grid(axis="x", color="#dddddd", linewidth=0.65)

        figure.legend(
            loc="upper center",
            bbox_to_anchor=(0.52, 0.98),
            ncol=3,
            frameon=False,
            fontsize=9,
            handletextpad=0.35,
        )

        wide = rows[index["wide"]]
        before = 100 * float(wide["signed-exact"])
        after = 100 * float(wide["split-exact"])
        axis.annotate(
            f"{before:.2f}% → {after:.2f}%",
            (after, index["wide"]),
            xytext=(-15, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.5,
            arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.7},
        )
        tiny = rows[index["tiny"]]
        for mode, position, precision, label, offset, alignment in (
            ("split-exact", 0, 3, "量化参考", (13, 0), "left"),
            ("sign-split", 0.24, 2, "模拟器", (-14, 0), "right"),
        ):
            value = 100 * float(tiny[mode])
            axis.annotate(
                f"{label} {value:.{precision}f}%",
                (value, index["tiny"] + position),
                xytext=offset,
                textcoords="offset points",
                ha=alignment,
                va="center",
                fontsize=8.5,
                arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.7},
            )

        figure.savefig(_FIGURES / "bf16-dot-error.png", dpi=220)


if __name__ == "__main__":
    main()
