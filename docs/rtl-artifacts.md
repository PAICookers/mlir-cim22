# INT8 RTL Artifact Contract

This exporter targets the supplier `CIM22_tb/tb_noc_20_cores_edge.sv` replay
profile. The 2026-09-05 model-team feedback identified incorrect CIM body
packing, a return route terminating inside the chip, and full-chip multicast
for a four-core model. These corrections do not change numeric lowering or
the compiler's general zero-Copy packet profile.

## Fields and Recipients

- A CIM body carries 32-bit data in `[39:8]` and its address in `[7:0]`.
  The exporter preserves the supplier multicast task's `0010` type nibble and
  zeros `[59:40]`. The generic codec retains its canonical zero high bits;
  both share only the payload packing routine. Body routing follows the head.
- Coordinates are `(row,col)`, with `core = row * 5 + col`. The injection point
  is `(0,0)`; the receiver is beyond `(0,4)` on Y-, at `(-1,4)`.
- Each responding core receives one onecast `1010` configuration before any
  reads. `[22:18]` contains the full five-bit core index. Its return fields are
  `(XY,X,Y) = (0,4-col,-1-row)`, encoded as six-bit sign and magnitude (`-1=33`).
  Reversing the ingress route or returning to self is not an off-chip route.
- Configuration and reads use each Macro's mapped recipient set. Work starts
  each participating core once. Contiguous same-row recipients with equal
  payloads share a positive Copy-X packet; holes and row boundaries split
  packets. No bounding rectangle may include an inactive recipient.
- Macro 0 and 1 remain separate configuration/read phases. A read-enabled Macro
  receives eight Cache requests and one 256-word CIM request. Existing
  equal-weight and dual-Macro export restrictions are unchanged.

The default four-core model uses base Core `(0,0,0)` and Copy `(0,3,0)`; it
still has 792 configuration, 2 work, and 20 readback flits. These counts alone
cannot distinguish the repaired frames from the original incorrect artifact.

## Reproduction and Evidence

`test/Integration/ONNX/rtl-artifact-export.test` generates the default model and
exports `01_config.frames.txt`, `02_work.frames.txt`, `03_readback.frames.txt`,
`sources/`, `expected/`, and a generated `README.md` with the actual mapping.
`rtl-artifact-targets.test` exercises a row boundary and response indices 16..19.
The independent Python checker decodes delivered fields and compares all 256
CIM addresses and data words, input Cache contents, return destinations and
indices, and exact per-phase recipient sets. It does not call the C++ encoder.

Before the fix, separate checks of the original four-core artifact failed on
duplicate CIM address 168, a return destination other than `(-1,4)`, and
configuration reaching inactive cores. The source model and numeric expected
results are not changed to accommodate frame output.

Replay the three delivered files without modifying them or injecting repair
configuration in the TB. Retain the TB's compute/readback synchronization.
Acceptance requires all input flits to be acknowledged, expected responses
without X values, complete CIM snapshots matching sources, correct Cache
results, and no sustained backpressure. Software field checks are necessary
but do not establish these RTL execution properties.

The frozen `9_5_model/rtl_CIM_CHIP` and `sim/REPORT_RAW_DEMO.md` are not in this
checkout. The original feedback reports sustained backpressure and zero
off-chip responses, but does not locate the first internal blocking node.
Do not claim that node, a new RTL PASS, or board deployability from this fix.
