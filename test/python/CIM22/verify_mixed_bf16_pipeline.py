"""Check BF16 boundary rounding and the outlined Host/kernel call graph on CPU.

Device kernels execute as standard Linalg reference functions here. This does
not execute packets, bind CIMRunner, or emulate the hardware BF16 algorithm.
"""

import argparse
from pathlib import Path
import re
import subprocess


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def module_body(text, name):
    """Read a printed test module using its unambiguous two-space nesting."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith(f"  module @{name} "))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "  }")
    return lines[start + 1:end]


def link_transaction_reference(text):
    """Link packet-stage Host wrappers to mathematical tile reference functions.

    Read the original BF16 static weights and actual configure/readback bindings,
    not the hardware's packed mantissas or a duplicated packet implementation.
    """
    host = module_body(text, "__cim_host")
    device = module_body(text, "__cim_device")
    weights = {}
    for line in device:
        match = re.match(r"\s*cim.static_weight @(\w+) = (.*) : tensor<16x64xbf16>$", line)
        if match:
            weights[match[1]] = match[2]
    functions = []
    for index, header in enumerate(device):
        if not header.startswith("    func.func @"):
            continue
        end = next(i for i in range(index + 1, len(device)) if device[i] == "    }")
        body = device[index + 1:end]
        args = re.findall(r"(%\w+): tensor<64xbf16>", header)
        block = next(line for line in body if "^bb0(" in line)
        block_args = re.findall(r"(%\w+): tensor<64xbf16>", block)
        if len(args) != len(block_args):
            raise AssertionError("transaction input signature mismatch")
        inputs, resources, results = {}, {}, {}
        for line in body:
            work = re.search(r"work_id = (\d+) : i64", line)
            if "cim.configure_input " in line:
                value = re.search(r"cim.configure_input (%\w+)", line)[1]
                inputs[work[1]] = args[block_args.index(value)]
            elif "cim.configure_weight " in line:
                resources[work[1]] = re.search(r"cim.configure_weight @(\w+)", line)[1]
            elif "cim.readback " in line:
                results[re.search(r"(%\w+) = cim.readback", line)[1]] = work[1]
        yield_line = next(line for line in body if '"cim.yield"' in line)
        returned = re.findall(r"%\w+", yield_line.split(") :")[0])
        # Preserve each external declaration's exact tensor signature.
        signature = header.split(" attributes ")[0].strip()
        signature = signature.replace("func.func @", "func.func private @")
        lines = [signature + " {", "  %zero = arith.constant dense<0.0> : tensor<16xbf16>"]
        for i, value in enumerate(returned):
            work = results[value]
            lines += [f"  %w{i} = arith.constant {weights[resources[work]]} : tensor<16x64xbf16>",
                      f"  %r{i} = linalg.matvec ins(%w{i}, {inputs[work]} : tensor<16x64xbf16>, tensor<64xbf16>) outs(%zero : tensor<16xbf16>) -> tensor<16xbf16>"]
        lines += ["  return " + ", ".join(f"%r{i}" for i in range(len(returned)))
                  + " : " + ", ".join("tensor<16xbf16>" for _ in returned), "}"]
        functions.extend(lines)
    if not functions:
        raise AssertionError("packet program has no transaction entry points")
    host = [line for line in host if not re.match(
        r"\s*func.func private @__cim_kernel_\d+_transaction_\d+\(", line)]
    aliases = [line for line in text.splitlines() if line.startswith("#")]
    return "\n".join(aliases + ["module {"] + host + functions + ["}"])


MAIN = """
func.func @main() -> i32 {
  // BF16 nearest-even rounds activation to 1 and pointwise rows to 1, 2, 3.
  %input = arith.constant dense<1.0009765625> : tensor<1x2x3x3xf32>
  %residual = arith.constant dense<[[0.25, -0.5]]> : tensor<1x2xf32>
  %actual = func.call @mixed(%input, %residual) : (tensor<1x2x3x3xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %x = tensor.extract %actual[%c0, %c0] : tensor<1x2xf32>
  %y = tensor.extract %actual[%c0, %c1] : tensor<1x2xf32>
  // PW=2,4,6; bias=2.5,4.5,6.5; DW=23.5,41.5,59.5; Linear=47,137.
  %expected_x = arith.constant 47.25 : f32
  %expected_y = arith.constant 136.5 : f32
  %same_x = arith.cmpf oeq, %x, %expected_x : f32
  %same_y = arith.cmpf oeq, %y, %expected_y : f32
  %ok = arith.andi %same_x, %same_y : i1
  // Also execute the runtime-weight FP32 fallback through the Host pipeline.
  %a = arith.constant dense<1.0> : tensor<2x3xf32>
  %w = arith.constant dense<0.5> : tensor<3x2xf32>
  %host = func.call @host_matmul(%a, %w) : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  %host_value = tensor.extract %host[%c0, %c1] : tensor<2x2xf32>
  %expected_host = arith.constant 1.5 : f32
  %host_ok = arith.cmpf oeq, %host_value, %expected_host : f32
  %host_and_mixed = arith.andi %ok, %host_ok : i1
  %cross_input = arith.constant dense<1.0> : tensor<65xf32>
  %cross = func.call @cross_k(%cross_input) : (tensor<65xf32>) -> tensor<1xf32>
  %cross_value = tensor.extract %cross[%c0] : tensor<1xf32>
  %expected_cross = arith.constant 65.0 : f32
  %cross_ok = arith.cmpf oeq, %cross_value, %expected_cross : f32
  %all_ok = arith.andi %host_and_mixed, %cross_ok : i1
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %status = arith.select %all_ok, %zero, %one : i32
  return %status : i32
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runner-utils", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stream-columns", type=int, default=1)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Check every output column against fixed scalar expectations, not just a
    # spatial sum that could hide a permutation of the streamed columns.
    column_check = [
        '  %columns = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 0.0, -1.0, -2.0]]> : tensor<2x5xbf16>',
        '  %ordered = func.call @stream_order(%columns) : (tensor<2x5xbf16>) -> tensor<2x5xbf16>',
        '  %order_ok0 = arith.constant true',
    ]
    for i, expected in enumerate((5, 4, 3, 2, 1, 1, -1, -3, -5, -7)):
        column_check += [
            f'  %row{i} = arith.constant {i // 5} : index',
            f'  %col{i} = arith.constant {i % 5} : index',
            f'  %value{i} = tensor.extract %ordered[%row{i}, %col{i}] : tensor<2x5xbf16>',
            f'  %expected{i} = arith.constant {expected}.0 : bf16',
            f'  %eq{i} = arith.cmpf oeq, %value{i}, %expected{i} : bf16',
            f'  %order_ok{i+1} = arith.andi %order_ok{i}, %eq{i} : i1',
        ]
    column_check.append('  %ordered_all_ok = arith.andi %all_ok, %order_ok10 : i1')
    main = MAIN.replace('  %zero = arith.constant 0 : i32',
                        '\n'.join(column_check) + '\n  %zero = arith.constant 0 : i32')
    main = main.replace('arith.select %all_ok,', 'arith.select %ordered_all_ok,')
    source = args.work_dir / "source.mlir"
    source.write_text(args.fixture.read_text(encoding="utf-8").replace(
        "dense<1.0> : tensor<3x2x1x1xf32>",
        "dense<[[[[1.0009765625]], [[1.0009765625]]], "
        "[[[2.0009765625]], [[2.0009765625]]], "
        "[[[3.0009765625]], [[3.0009765625]]]]> : tensor<3x2x1x1xf32>")
        + main, encoding="utf-8")
    prepared = args.work_dir / "prepared.mlir"
    run([args.opt, str(source), "--cim-bf16-prepare", "-o", str(prepared)])
    split = run([args.opt, str(source), f"--cim-bf16-pipeline=stage=split max-kernel-columns={args.stream_columns}"])
    assert "scf.for" in split and "tensor.insert_slice" in split
    host = module_body(split, "__cim_host")
    device = module_body(split, "__cim_device")
    declarations = [line for line in host if re.match(
        r"\s*func.func private @__cim_kernel_\d+\(", line)]
    if len(declarations) != 7:
        raise AssertionError("expected pointwise, three depthwise, Linear, cross-K and column-order kernels")
    # Link the exact outlined standard-Linalg definitions for a CPU-only oracle.
    host = [line for line in host if line not in declarations]
    aliases = [line for line in split.splitlines() if line.startswith("#")]
    linked = args.work_dir / "linked-reference.mlir"
    device = [line.replace("func.func @__cim_kernel_", "func.func private @__cim_kernel_")
              for line in device]
    linked.write_text("\n".join(aliases + ["module {"] + host + device + ["}"]),
                      encoding="utf-8")
    packet = run([args.opt, str(source), f"--cim-bf16-pipeline=max-kernel-columns={args.stream_columns}"])
    transaction_reference = args.work_dir / "transaction-reference.mlir"
    transaction_reference.write_text(link_transaction_reference(packet), encoding="utf-8")
    for path, expected in ((source, "1"), (prepared, "0"), (linked, "0"),
                           (transaction_reference, "0")):
        lowered = path.with_suffix(".llvm.mlir")
        run([args.opt, str(path), "--cim-host-to-llvm", "-o", str(lowered)])
        result = run([args.runner, str(lowered), "-e", "main",
                      "-entry-point-result=i32", "-shared-libs=" + args.runner_utils])
        if result.strip() != expected:
            raise AssertionError(f"{path.name}: expected {expected}, got {result}")
    print("PASS mixed BF16 pipeline: rounding, Host fallback and outlined calls")


if __name__ == "__main__":
    main()
