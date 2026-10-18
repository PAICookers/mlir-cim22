# CIM22 模型数据 Schema

本目录的 [cim22_model.proto](cim22_model.proto) 定义编译器与原生 runtime
交换的静态模型数据：边界张量签名、CIM kernel、执行计划、权重与绑定。
采用 `proto3`，包名为 `cim22.artifact.v1`，C++ 使用 `LITE_RUNTIME`。
本文以当前实现为准；字段能被解析、模型能被加载和板端能执行是不同层面的契约。

## 数据在哪里，谁负责读写

当前 AOT 编译把 Protobuf 字节和 Host 原生代码一起嵌入模型共享库。
导出函数 `cim22_model_v1()` 返回 `CIM22ModelV1`，其中 `data/data_size`
指向 Protobuf，`invoke` 指向 Host 入口。当前加载接口接收模型共享库，
不是独立的 `manifest.pb` 文件。

`CIM22ModelV1.abi_version` 标识共享库描述符布局及 `invoke(void **)` 的原生
调用约定，runtime 当前要求为 1；`schema_version` 标识 `data` 内的 Protobuf
格式。修改 schema 字段不必升级调用 ABI；修改描述符布局或调用约定时另行评估。

| 环节 | 实现与职责 |
| --- | --- |
| 生成执行计划 | [CIMExecutable.cpp](../lib/Target/CIMExecutable.cpp)：生成 `CIMTransaction`、静态 packet 与 flit 数据 |
| 内存表示 | [CIMExecutable.h](../include/CIM22/Execution/CIMExecutable.h)、[ModelData.h](../lib/Runtime/ModelData.h)：拥有静态数据，不保存活动设备或输入缓冲区 |
| 编解码与加载检查 | [ModelData.cpp](../lib/Runtime/ModelData.cpp)：`encodeModelData` / `decodeModelData`、`validate` / `validateTransaction` |
| 嵌入模型 | [Compiler.cpp](../lib/Runtime/Compiler.cpp)：生成 dispatch wrapper，嵌入数据和描述符 |
| 原生 ABI 与加载 | [ModelABI.h](../include/CIM22/Execution/ModelABI.h)、[Runtime.cpp](../lib/Runtime/Runtime.cpp)：检查 ABI，解码数据，按 backend 创建或配置 runner |
| 执行 | [CIMSoftwareRunner.cpp](../lib/Execution/CIMSoftwareRunner.cpp)、[CIMBoardRunner.cpp](../lib/Runtime/CIMBoardRunner.cpp)：执行语义或组装实际板端程序，并施加额外约束 |
| 代码生成 | [Runtime/CMakeLists.txt](../lib/Runtime/CMakeLists.txt)：生成 C++ 与 Python 测试绑定 |

Schema 不保存活动张量、设备句柄、USB 配置、线程/锁或会话执行状态。
它也不定义 USB record 格式、NoC 位域编码或 Host 函数调用约定。
更改这些契约需检查各自实现，不能只增加一个 Protobuf 字段。

## 字段说明

表中的数字是 Protobuf 字段号；范围和必填条件指当前加载器的行为，
不是 `.proto` 类型系统自动提供的检查。

### ModelArtifact 与边界签名

| 字段 | 含义与约束 |
| --- | --- |
| `schema_version = 1` | 整份模型 schema 的唯一持久化版本号，覆盖签名与所有 transaction 数据。开发期写出 `1`；当前 reader **不检查**其值，缺失或其他值也可能通过，尚无严格版本协商 |
| `inputs = 2` / `outputs = 3` | 有序边界 `TensorSignature`；按 Host 参数/结果顺序保存。它们不同于 transaction 内部的输入与回读绑定 |
| `kernels = 4` | 有序 `Kernel`；列表位置就是编译进 Host 代码的 native dispatch index，不能独立排序或重排。纯 Host 模型可为空 |
| `backend = 5` | `SOFTWARE = 0`、`USB = 1`；字段缺失时按 proto 默认值成为软件模式，未知枚举值被拒绝。编译器会显式选择；编译选项当前默认 USB，和字段缺失时的默认值不同 |

`TensorSignature.dimensions = 1` 是静态维度列表：空列表表示 rank-0 标量，
不是未知 shape。各维必须大于 0，元素总数不能溢出 `int64_t`。
`data_type = 2` 在边界只能为 `FP32`；CPU、连续布局等内存要求由调用边界检查，
schema 本身没有 stride/device 字段。

`Kernel.kernel_name = 1` 必须非空且在模型内唯一，用于标识与错误定位；
`transaction = 2` 必须存在。运行时按列表索引 dispatch，并非按名字动态查找 kernel。

`DataType` 的固定编号为 `UNSPECIFIED = 0`、`INT8 = 1`、
`BF16 = 2`、`FP32 = 3`。当前 transaction 的权重、输入、回读与 packet
只允许 `INT8` 或 `BF16`；边界只允许 `FP32`。枚举中存在某类型不意味着所有路径支持它。

### Transaction：一个 kernel 的静态执行数据

| 字段 | 含义与约束 |
| --- | --- |
| `target_profile = 1` | 当前必须为 `cim22-4x5-v1` |
| `groups = 2` | 分组与 work 放置信息；整个 transaction 必须有 work |
| `weights = 3` | 每 work 一项静态权重 |
| `packets = 4` | 语义 packet 元数据，含绑定、路由和回读序列；不是实际 USB 发包顺序的完整记录 |
| `inputs = 5` | 每 work 一项动态输入绑定，列表顺序决定 runner 输入 view 的顺序 |
| `readbacks = 6` | 每 work 一项回读绑定，列表顺序决定 dispatch 输出的顺序 |
| `flits = 7` | `packed fixed64` 静态编码 flit 序列，必须非空；加载器不解析其内容，也不验证其与 packet/权重一致 |

当前 software runner 使用语义绑定和权重计算，USB runner 根据绑定重新组装程序；
两者都不直接执行 `flits`。静态 flits 不包含本次调用的动态输入，
不能直接当作完整 USB 发送流。`packets` 与 `flits` 也不是逐元素一一对应关系。

### Group 与 Work：身份和放置

| 字段 | 含义与约束 |
| --- | --- |
| `Group.group_id = 1` | 显式存在、非负，transaction 内唯一 |
| `Group.works = 2` | 不能空 |
| `Work.work_id = 1` | 显式存在、非负，与 `group_id` 组成唯一逻辑键 |
| `Work.core_slot = 2` | 显式存在，范围 `[0, 19]` |
| `Work.macro_slot = 3` | 显式存在，范围 `[0, 1]`；同 group 内 `(core_slot, macro_slot)` 不能重复 |
| `Work.route = 4` | 恰好 6 个 `sfixed32` 路由分量 |

当前路由表示的前三项为 diagonal XY、剩余 X、剩余 Y，后三项为 Copy 路由。
生产构造与 codec 要求分量在 `[-31, 31]`，当前 Copy 三项为 0；
**ModelData 加载器仅检查长度和绑定间一致性，不检查这些数值约束**。
具体编码由 [CIMPacketCodec.cpp](../lib/Support/CIMPacketCodec.cpp) 与
[CIMExecutable.cpp](../lib/Target/CIMExecutable.cpp) 负责，不能从字段名另造位域规则。

### StaticWeight：同一权重的执行表示

| 字段 | 含义与约束 |
| --- | --- |
| `group_id = 1` / `work_id = 2` | 显式存在，必须引用已有 work；每 work 只能有一项 |
| `core_slot = 3` / `macro_slot = 4` | 显式存在，与对应 work 放置一致 |
| `resource = 5` | 非空 MLIR 权重资源符号名；不是外部文件路径，runner 不据此加载文件 |
| `route = 6` | 长度 6，与对应 work 的 route 一致 |
| `words = 7` | 256 个 packed 原始 32-bit word 的位模式；`sfixed32` 的符号不等于单个数值权重的符号 |
| `values = 8` | 1024 bytes，即 `16 × 64` 个有符号字节位模式；软件按 `values[lane * 64 + k]` 使用。BF16 模式下保存整数尾数表示，不是原始 16-bit BF16 权重数组 |
| `exponents = 9` | 16 bytes，每输出 lane 一个无符号指数；INT8 也必须满足该长度 |
| `data_type = 10` | `INT8` 或 `BF16`，与同 work 的输入/回读及关联 packet 一致 |

软件路径使用 `values/exponents`，USB 程序组装使用 `words/exponents`。
加载器检查长度，但**不会重算三者的内容一致性**。维护权重布局时应复用
[Int8WeightLayout.cpp](../lib/Support/Int8WeightLayout.cpp) 和现有构造逻辑，
同步验证两条路径；不要只改其中一种表示或手写第二套布局算法。

### InputBinding 与 ReadbackBinding

两者的 `group_id = 1`、`work_id = 2`、`macro_slot = 3` 都必须显式存在，
引用已知 work，并与其 macro 放置一致；每 work 各有且只有一项。

| 消息与字段 | 含义与约束 |
| --- | --- |
| `InputBinding.input_slot = 4` | 显式存在、非负；编译生成的 Host stub 参数索引。不是整个模型的公开输入索引，也不是 Cache 地址；编译 wrapper 还检查其不越过 stub 参数数量 |
| `InputBinding.data_type = 5` | `INT8` 或 `BF16`，与同 work 权重一致 |
| `InputBinding.input_cache_address = 6` | 输入 Cache 地址 `[0, 7]`；USB 必须显式存在，软件模式允许缺失并取 proto 默认 0 |
| `ReadbackBinding.output_cache_address = 4` | 显式存在，输出 Cache 地址 `[0, 7]` |
| `ReadbackBinding.route = 5` | 长度 6，与对应 work 路由一致 |
| `ReadbackBinding.test_core = 6` | 长度 3，顺序为 `test_core_xy, test_core_x, test_core_y`，描述返回 test host 的路由距离；加载器不验证其数值范围 |
| `ReadbackBinding.data_type = 7` | `INT8` 或 `BF16`，与同 work 权重一致 |

### Packet 与 PacketKind

| 字段 | 含义与约束 |
| --- | --- |
| `kind = 1` | 已知且非 `PACKET_KIND_UNSPECIFIED` 的枚举值 |
| `route = 2` | 长度 6；绑定 work 的非 `CONTROL`、非 `WORK` packet 要与其权重或回读 route 一致 |
| `group_id = 3` / `work_id = 4` | 必须显式存在；未使用时保留 `-1`，不能省略后变成 0。非负时需引用已有 group/work |
| `macro_slot = 5` | 必须显式存在，范围 `[-1, 1]`；绑定 work 时需一致，`-1` 表示未使用 |
| `cache_address = 6` | 必须显式存在，范围 `[-1, 7]`；`-1` 表示未使用；输出读取时与回读绑定地址一致 |
| `test_core = 7` | 长度 3；`RETURN_ROUTE` 时必须与对应回读绑定一致 |
| `data_type = 8` | `INT8` 或 `BF16`；引用 work 时与该 work 权重一致 |

| 枚举编号 | 用途 |
| --- | --- |
| `CONTROL = 1` | 控制配置 |
| `WEIGHT = 2` | 权重配置 |
| `WEIGHT_EXPONENT = 3` | BF16 权重指数配置 |
| `WORK = 4` | group 工作命令 |
| `INPUT_CACHE_WRITE = 5` | 动态输入写 Cache |
| `RETURN_ROUTE = 6` | 回传路由配置 |
| `OUTPUT_CACHE_READ = 7` | 读取输出 Cache |

当前加载器要求每 work 出现 `WEIGHT`、`INPUT_CACHE_WRITE`、`RETURN_ROUTE`、
`CONTROL`、`OUTPUT_CACHE_READ`，BF16 还需 `WEIGHT_EXPONENT`。
`WORK` packet 必须指向 group；除 `CONTROL`、`WORK` 外必须绑定 work。
每个 `RETURN_ROUTE` 后必须紧接同绑定、同路由、同类型的
`CONTROL → OUTPUT_CACHE_READ`。这些检查不等于完整验证全局执行顺序；
加载器也没有要求每 group 必须存在 `WORK` packet。

## 维护与扩展

### 先确定改变哪一层

1. 只改说明：核对 `.proto` 和实际 producer/consumer，更新本 README；无需重生成绑定或重装包。
2. 加描述性字段：决定缺失值和旧 reader 忽略它时是否仍安全，再实现读写与测试。
3. 改执行语义、布局或类型：同时审查 transaction 构造、native 数据结构、
   encoder/decoder/validator、Host wrapper、software/USB runner 及相关 codec。
   新字段即使 wire-compatible，也可能不能由旧 runtime 安全执行。
4. 改持久化契约时统一评估顶层 `schema_version`。只有实际改变 MLIR/native
   transaction 契约或调用 ABI 时，才同步它们各自的内部版本和
   `CIM22ModelV1.abi_version`；不要重新引入无独立演进用途的子版本号。

开发期顶层版本不强制校验是当前事实，不是永久兼容保证。
若开始强制校验，需一起调整 writer、reader、兼容策略与
`test_development_metadata_contract`，并同步项目规则。

### Protobuf 演进规则

- **项目规则：当有字段删除时，如无显式要求，无需预留字段，且重新紧凑编号。**
  该规则适用于整份 schema 的各个消息，不限于 `Transaction`。
  默认不添加 `reserved`，剩余字段按原顺序从 1 连续编号。必须同时更新生成绑定、
  读写代码、测试与字段表，并重新编译模型；重新编号不保证旧产物兼容。
- 只有用户显式要求保留字段号、预留字段或兼容旧格式时，才按该要求采用
  固定编号、`reserved` 或迁移方案。不要自行引入字段预留。
- 默认新增字段使用新编号；明确其单位、范围、顺序、presence 和缺失行为。
  不随意改变现有字段的类型、标量/repeated 形态或身份字段的 `optional`。
- `optional` 不等于“业务上可省略”：work/绑定的槽位 0 合法，必须用 `has_*`
  区分缺失。packet 的 `-1` 哨兵应显式写出，不能被默认 0 替代。
- 新枚举值必须同步 native enum、转换 switch、validator 和使用方。
  当前未知 backend、datatype、packet kind 会被拒绝，不存在自动降级保证。
- 保持 repeated 的语义顺序，尤其 kernel dispatch、输入 view、回读及局部 packet
  序列；不要换成 map 或统一排序。
- 未知描述字段可以被忽略，但新执行字段不能依赖旧 reader 的忽略行为。
  Protobuf 字节也不是跨版本的规范化表示，不应用序列化字节等价替代语义兼容判断。

通用规则依据：官方 [proto3 演进说明](https://protobuf.dev/programming-guides/proto3/)、
[字段 presence](https://protobuf.dev/programming-guides/field_presence/)、
[序列化非规范化说明](https://protobuf.dev/programming-guides/serialization-not-canonical/)。
项目实际接受什么，仍以 reader 和测试为准。

本次开发调整还沿用了顶层字段的重新编号与 `backend` 命名，未提供旧模型迁移。
旧共享库必须重新编译；当前顶层版本尚未强制检查，不能依赖它自动识别旧布局。

### 生成与验证

依赖由 [MLIRCIM22Dependencies.cmake](../cmake/modules/MLIRCIM22Dependencies.cmake)
固定为 Protobuf/protoc 31.1 配对，runtime 链接 `protobuf::libprotobuf-lite`。
不要切换成系统 `protoc` 或另加 Buf/gRPC 等构建路径。
C++ 和 Python 绑定生成在所选 CMake build 目录的 `lib/Runtime/generated/` 下；
Python 仓库副本只用于测试，位于
[`python/tests/_generated/cim22_model_pb2.py`](../python/tests/_generated/cim22_model_pb2.py)，
由目标复制更新，不手改。Python 测试依赖见
[`python/pyproject.toml`](../python/pyproject.toml)。

以下命令在仓库根执行，示例沿用 [BUILDING.md](../BUILDING.md#测试与开发)
已配置的原生测试构建 `build/test`。使用其他构建目录时替换路径，
保留原有工具链、构建类型和 USB/ONNX 等选项。

```sh
# .proto 有变化时，更新仓库中的 Python 测试绑定。
cmake --build build/test --target update-cim22-protobuf
# 比较新生成绑定与仓库副本；只读检查不需要执行 update。
cmake --build build/test --target check-cim22-protobuf
```

修改 `.proto` 或 native 实现后，还需按 BUILDING.md 重新构建安装 editable 包：

```sh
uv sync --project python --python 3.12 --locked --group dev \
  --reinstall-package mlir-cim22
uv run --directory python --no-sync pytest \
  tests/test_model_format.py tests/test_artifacts.py
```

上面的 sync 使用默认构建配置；若已有自定义配置，需携带原来的 `-C` 参数。
重建后重启加载原生库的进程并重新编译模型，不能拿旧 `.so` 假定符合新 schema。
不要在另一个进程仍加载/链接同一组产物时并行重装。

验证按变更范围选择：

- 格式/加载：[test_model_format.py](../python/tests/test_model_format.py)
  覆盖未知字段、缺失/损坏数据、版本/backend、presence、绑定、长度与回读序列；
  它用测试 DSO 验证加载，不代表完成模型数值执行。
- 编译产物与边界：[test_artifacts.py](../python/tests/test_artifacts.py)、
  [test_native_compiler.py](../python/tests/test_native_compiler.py)。
- 执行计划、布局、packet 或 USB 变更：再运行对应原生 lit/unit 和
  [test_usb_runtime.py](../python/tests/test_usb_runtime.py) 中相关用例。
  原生测试构建选项与完整命令见 BUILDING.md；软件或 fake device 结果不能写成真板验证。
- 仅文档/技能变更：检查本地链接、源代码一致性、技能 validator 和
  `git diff --check`；无需引入新的测试、脚本或依赖。

## 使用时的边界

- `ModelData` 校验覆盖格式、固定长度和绑定引用，不是完整的 target/runner verifier。
  USB runner 还检查 core/group、路由范围和返回路由等额外限制；不要只靠
  “Protobuf 解析成功”判断可执行性。
- schema 接受 `BF16` 不等于板端 BF16 数值回读已闭环。当前 board runner 的
  decoded BF16 `run` 仍拒绝执行，raw `capture` 属诊断路径；应查看其当前实现与未决契约。
- 加载模型共享库会经过 `dlopen` 和模型导出函数；Protobuf 校验不能使不可信原生库安全。
  模型仍需来自可信编译流程。
- 修改软硬件语义时使用现有项目证据与文档技能；本 README 只描述软件契约，
  不替代芯片规范、传输协议或板端验证记录。
