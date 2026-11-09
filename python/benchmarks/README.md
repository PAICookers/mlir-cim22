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

# BF16 点积数值基准

`bf16_dot_benchmark` 使用无偏置的 `64 → 16` 矩阵向量乘。每组策略使用相同的 BF16
输入与权重，通过 Linalg 编译到生产模拟器。默认种子为 20261106，每个分布或指数扫频
用例有 128 个输入向量、16 个固定权重通道。三个定向用例各有一个输入向量。

| 用例 | 测量条件 |
| --- | --- |
| `positive`、`signed` | 幅值在 `[1, 2)`，分别采用非负及混合符号操作数 |
| `wide` | 混合符号，组内指数在 `[-24, 0]` 取样，测试小项被共享尺度截掉 |
| `sparse` | 输入和权重独立保留约八分之一元素，含全零输入及无交集的非零位置 |
| `cancellation` | 大项精确相消，剩余项从零到带符号的 `2^-12 … 1` |
| `tiny`、`large` | 扫描输入尺度，覆盖非正规操作数、输出下溢及高指数的有限结果 |
| `zero-collision` | 8 个真正的零整数部分和，另 8 个整数部分和为 128，检查回包值冲突 |
| `overflow`、`overflow-cancel` | 分别覆盖点积超出 BF16 有限范围、精确点积为零但单个分支超范围 |

从 `python/` 运行，输出目录须为新目录。可重复指定 `--case` 运行子集，
`--samples` 控制各分布或扫频的输入向量数。编译及 Session 调用依次进行。

```sh
uv run --no-sync python -m benchmarks.bf16_dot_benchmark \
    --output-dir ../build/validation/bf16-dot-benchmark
```

`summary.json` 与 `summary.csv` 记录各用例指标。每个用例目录保存 Linalg、三种
制品和 `sample.npz`，可重放实际输入和权重并核对输出。NPZ 中被拒绝调用的输出
位置用 NaN 占位，由独立的 `*_rejected` 掩码标记。报告中的 NaN 计数仅来自成功调用。

- `none`、`legacy`、`sign-split` 为实际执行路径。`legacy` 是项目内复现的供应方
  尺度补偿思路。`signed-exact` 对有符号操作数分别使用一个共享尺度，按幅值截断后恢复
  符号，再精确累加。`split-exact` 让正负幅值各选尺度，同样精确累加。这两条数学参考线
  都省略了转换器截断及输出舍入，适用于分析量化误差。
- 精确参考点积以 `2^-266` 为单位做整数累加。归一化绝对误差为
  `abs(output - exact) / sum(abs(x[i] * w[i]))`。分母为零时只检查零值。
  误差分位数只统计精确结果在 BF16 有限范围内且实际结果有限的位置。比较
  `legacy` 与 `sign-split` 时，可用两者共同有效位置的 P99。常规相对误差按
  `sum(abs(x[i] * w[i])) / abs(exact)` 分桶。另报 `sign-split` 到
  `split-exact` 的归一化差值，用于观察量化后的截断及舍入。
- NaN、正负 Inf、拒绝调用和实际 CIM 调用数分别统计。一次调用若被拒绝，16 个
  输出都不可用。调用数只累加成功调用。非零参考值被算成零时，另计其中按 BF16
  最近偶数舍入本来就会归零的数量。

默认样本的一次模拟器实测如下。P99 列比较两条补偿路径共同有效的输出。

| 用例 | `none` 的 NaN/Inf | `legacy` 归一化误差 P99 | `sign-split` 归一化误差 P99 |
| --- | ---: | ---: | ---: |
| `positive` | 2,048 / 2,048 | 1.265% | 1.265% |
| `signed` | 2,046 / 2,048 | 0.206% | 0.344% |
| `wide` | 2,029 / 2,048 | 9,596.7% | 42.11% |
| `sparse` | 2,025 / 2,048 | 1.489% | 1.739% |
| `cancellation` | 1,354 / 2,048 | 49.81% | 0.585% |
| `tiny` | 0 / 2,048 | 392.58% | 32.57% |
| `large` | 2,048 / 2,048 | 0.206% | 0.344% |

这七组里，两条补偿路径都返回有限结果。稀疏组中 768 个精确零经
`sign-split` 全部返回零，`legacy` 全部返回非零。近抵消组里仍有符号翻转，
高精度要求应同时看抵消分桶。范围探针中，`sign-split` 对真正的溢出和单分支
溢出均报告错误。通用有符号 Linalg 入口实测每个输入调用 4 次 CIM，另两条
路径各调用 1 次。该计数对应这里的固定形状和编译入口。

两条数学参考线也能区分改善来自哪里。宽指数用例的 P99 从
`signed-exact` 的 45.01% 降到 `split-exact` 的 42.11%，表明单独给正负幅值
选尺度只消除了一部分量化误差。极小值用例的两条参考线均为 0.180%，但
`sign-split` 实测为 32.57%，此时应结合下溢计数和尺度恢复的舍入分析。
