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

USB 协议和 PLL 参数/手动时钟测试共用源码。Linux 执行 libusb 与 PTY 串口用例；Windows 执行 CyAPI 用例及自动串口不支持检查。平台专属用例使用 lit feature 或源码条件排除，不依赖缺失程序报错。

PLL 测试将生产实现与测试链接进同一可执行文件，避免 Windows 下 LLVM 错误对象跨 DLL。`MLIRCIM22_INCLUDE_TESTS=ON` 需要同时启用 ONNX；默认发布构建关闭这两个选项。

环境准备与运行命令见 [BUILDING](../BUILDING.md#测试与开发)。
