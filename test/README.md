# 测试体系

本目录供开发者和 AI agent 定位原生测试；按被测层次组织：

| 位置 | 职责 |
| --- | --- |
| `Dialect/`、`Conversion/`、`Frontend/`、`Target/` | 对应生产模块的 IR、转换、导入和目标输出测试 |
| `Integration/` | 跨阶段编译流水线与数值验证 |
| `Unit/` | C++ API 单元测试，由 lit 包装调用 |
| `python/` | 独立数值 oracle、fixture 解析和产物检查，由 `.test` 包装调用 |
| `Inputs/` | 不可变测试输入与参考数据，不参与 lit 测试发现 |
| [../python/tests/](../python/tests/) | 独立的 pytest 测试，覆盖 Python 后端、原生接口及软件执行 |

原生测试由 lit 统一组织；Python 包测试由 pytest 组织。新增测试放入对应层次，优先扩展已有测试。

环境准备与运行命令见 [BUILDING](../BUILDING.md#测试与开发)。
