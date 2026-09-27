"""Generic entry specialization and strict-offload contract tests."""
import argparse
from pathlib import Path
import subprocess

def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', required=True)
    parser.add_argument('--work-dir', required=True, type=Path)
    a = parser.parse_args()
    work = a.work_dir
    work.mkdir(parents=True, exist_ok=True)
    source = work / 'entry.mlir'
    source.write_text('''func.func @main(%x: tensor<2x3xf32>, %w: tensor<3x2xf32>) -> tensor<2x2xf32> {
      %z = arith.constant dense<0.0> : tensor<2x2xf32>
      %r = linalg.matmul ins(%x, %w : tensor<2x3xf32>, tensor<3x2xf32>) outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>
      return %r : tensor<2x2xf32>
    }''')
    manifest = work / 'params.mlir'
    def bind(attrs):
        manifest.write_text('module attributes {cim.parameters = {main = {' + attrs + '}}} {}')
        return subprocess.run([a.opt, str(source), f'--bind-cim-parameters=file={manifest}'], capture_output=True, text=True)
    ok = bind('arg1 = dense<0.5> : tensor<3x2xf32>')
    assert ok.returncode == 0 and '%arg1:' not in ok.stdout
    bound = work / 'bound.mlir'
    bound.write_text(ok.stdout)
    result = run([a.opt, str(bound), '--cim-bf16-pipeline=stage=llvm require-cim=true max-kernel-columns=1'])
    assert 'cim.transaction' in result
    for attr in ('arg1 = dense<0.5> : tensor<2x3xf32>', 'arg2 = dense<0.5> : tensor<3x2xf32>',
                 'arg1 = dense<0.5> : tensor<3x2xf32>, arg01 = dense<0.5> : tensor<3x2xf32>'):
        assert bind(attr).returncode != 0
    rejected = subprocess.run([a.opt, str(source), '--cim-bf16-pipeline=require-cim=true'], capture_output=True, text=True)
    assert rejected.returncode != 0 and 'required CIM' in rejected.stderr
    # ABI specialization must never silently invalidate an internal caller.
    source.write_text(source.read_text() + '''
    func.func @caller(%x: tensor<2x3xf32>, %w: tensor<3x2xf32>) -> tensor<2x2xf32> {
      %r = func.call @main(%x, %w) : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
      return %r : tensor<2x2xf32>
    }''')
    assert bind('arg1 = dense<0.5> : tensor<3x2xf32>').returncode != 0

    print('PASS entry parameter types, specialization, callers and strict offload')

if __name__ == '__main__':
    main()
