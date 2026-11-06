# CIM 计算效率基准

基准使用确定性输入和常量权重，覆盖卷积、矩阵向量乘和矩阵乘法。每组对照固定用例、
输入、权重及 Release 工具链，每个编译变体在独立进程中运行。每个进程创建一个 Session，
预热一次、计时五次，记录原始值、中位数和范围。编译、制品加载和输出哈希计算单独处理。

## 用例矩阵

浮点用例以 FP32 输入输出，CIM 内部采用 BF16 sign-split。整数用例以有符号 INT8 输入
和权重、INT32 输出运行。尺寸中的 `×` 分隔各维。

| 卷积用例 | 精度 | 输入 → 输出 | 权重尺寸 | Kernel / groups | Stride / dilation / padding | 每组 K×N |
| --- | --- | --- | --- | --- | --- | --- |
| `depthwise-3x3` | BF16 | `1×4×8×8 → 1×4×8×8` | CHW `4×3×3` | 3×3 / 4 | 1 / 1 / 1 | 9×1 |
| `pointwise-small` | BF16 | `1×2×8×8 → 1×16×8×8` | FCHW `16×2×1×1` | 1×1 / 1 | 1 / 1 / 0 | 2×16 |
| `pointwise-tail` | BF16 | `1×32×3×11 → 1×16×3×11` | FCHW `16×32×1×1` | 1×1 / 1 | 1 / 1 / 0 | 32×16 |
| `grouped-dilated` | BF16 | `2×2×4×7×9 → 2×2×5×7×9` | GFCHW `2×5×4×3×3` | 3×3 / 2 | 1 / 2 / 2 | 36×5 |
| `{bf16,int8}-conv-strided` | BF16、INT8 | `1×8×11×13 → 1×20×6×7` | FCHW `20×8×3×3` | 3×3 / 1 | 2 / 1 / 1 | 72×20 |

分组卷积采用 NGCHW，其余采用 NCHW。Padding 以空间维两侧各补的零元素数表示。
`pointwise-tail` 有 33 个空间列。标准卷积同时跨越 64 项 K 分组和 16 个输出 lane，
用于观察空间复用、归约尾块与回读开销。

矩阵类用例使用权重 `[N,K]` 左乘输入 `[K,C]`，输出为 `[N,C]`。矩阵向量乘的
输入和输出分别为 `[K]` 和 `[N]`。GEMM 名称的末尾依次为 `KxNxC`。

| 用例 | 精度 | K×N×C | 覆盖条件 |
| --- | --- | --- | --- |
| `bf16-matvec-aligned` | BF16 | 256×512×1 | 对齐的 K、N，多波次和符号分支 |
| `bf16-matvec-tail` | BF16 | 257×130×1 | K 与 N 均有尾块 |
| `{bf16,int8}-gemm-64x16x{1,31,32}` | BF16、INT8 | 64×16×{1,31,32} | 单列对照、不满列块、恰好一列块 |
| `{bf16,int8}-gemm-65x17x{33,65}` | BF16、INT8 | 65×17×{33,65} | 多列块及尾块，K 与 N 均跨 tile |
| `{bf16,int8}-gemm-256x320x3` | BF16、INT8 | 256×320×3 | 每列的 80 个 Work 超出单波次的 40 槽，存在批次间的权重替换 |

## 对照方法

| 变体 | 列块 | FC/Linear 权重重排 |
| --- | ---: | --- |
| `column-1` | 1 | 关闭 |
| `column-32` | 32 | 关闭 |
| `column-auto` | 自动 | 关闭 |
| `resident-order` | 32 | 开启 |

比较 `column-1` 和 `column-32` 可隔离列块化，比较 `column-32` 和 `column-auto`
可评估自动选宽，比较 `column-32` 和 `resident-order` 可隔离编译重排。
各组均使用生产模拟器的运行时权重驻留。单列小工作集和卷积提供
对照，多波次 GEMM 用于观察相同权重批次相邻后的变化。

生成阶段先实际执行，再发布制品。INT8 另用独立 NumPy INT32 累加验证。测量阶段比较
预热和每次计时调用的输出哈希与实际 CIM dispatch 数。`--baseline` 比较输入与输出的
形状、数据类型和哈希，同时适用于 BF16 与 INT8。

## 运行方法

先按 [构建说明](../../docs/BUILDING.md) 安装 Release 构建及 Python dev 依赖。
以下示例从 `python/` 目录在 Linux 上执行。Windows 的制品后缀为 `.dll`。

```sh
uv run --no-sync python -m benchmarks.build_cases --case bf16-conv-strided \
    --variant column-1 --variant column-32 --output-dir /tmp/cim22-bench
case_dir=/tmp/cim22-bench/bf16-conv-strided
uv run --no-sync python -m benchmarks.work_merge_benchmark "$case_dir/column-1.so" \
    --sample "$case_dir/sample.npz" --output "$case_dir/column-1.json"
uv run --no-sync python -m benchmarks.work_merge_benchmark "$case_dir/column-32.so" \
    --sample "$case_dir/sample.npz" --baseline "$case_dir/column-1.json" \
    --output "$case_dir/column-32.json"
```

可重复指定 `--case` 或 `--variant`。省略时生成该维度的全部条目。每个用例目录
保留 Linalg IR、确定性输入 NPZ 和制品。重复测量可复用制品，重编译请使用新目录。
已有模型也可使用同一测量入口。传入 simulator 制品及按 Session 输入顺序保存的 NPZ。

## 结果口径

- `dynamic_dispatches` 来自实际执行。`static_plan` 对制品中的每个 kernel 计一次，
  含 Host 循环的制品需另行采集各 kernel 的调用次数才能计算动态总量。
- 静态计划记录 Work、WorkOnce、Core/Macro 槽位、配置批次、回读批次、请求和响应。
  USB 字节及固定等待按全量权重写入和当前 Macro 分批回读规则估算。
- 权重驻留写入次数及真实 USB 字节需由带计数的运行时采集。PLL 复用通过功能回归检查。
  此基准运行在软件模拟器上，计时和估算结果分别报告。
- 对比模拟器实现时固定制品和样本，只切换运行时构建。报告中的制品、运行时和输入哈希
  可用于核对控制变量。避免并行计时或在计时期间编译。
