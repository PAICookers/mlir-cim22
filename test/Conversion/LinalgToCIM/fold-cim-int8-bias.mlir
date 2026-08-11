// RUN: mlir-cim22-opt %s -fold-cim-int8-bias > %t.once
// RUN: mlir-cim22-opt %t.once -fold-cim-int8-bias > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s --check-prefix=DEFAULT < %t.once
// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(fold-cim-int8-bias{allow-extra-k-tile=false})' | FileCheck %s --check-prefix=NOEXTRA
// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(fold-cim-int8-bias{enable=false})' | FileCheck %s --check-prefix=DISABLED

// DEFAULT-LABEL: func.func @linear_tail
// DEFAULT: arith.constant dense<1> : tensor<1x1xi8>
// DEFAULT: arith.constant dense<{{.*}}> : tensor<2x64xi8>
// DEFAULT: tensor.concat dim(0) {{.*}} -> tensor<64x1xi8>
// DEFAULT: linalg.matmul {cim.onnx.matmul_integer} ins({{.*}}tensor<2x64xi8>, tensor<64x1xi8>)
// DEFAULT-NOT: linalg.broadcast
// DEFAULT-NOT: linalg.add
// DEFAULT: return
func.func @linear_tail(%input: tensor<1x63xi8>) -> tensor<1x2xi32> {
  %weight = arith.constant dense<0> : tensor<2x63xi8>
  %input_empty = tensor.empty() : tensor<63x1xi8>
  %normalized_input = linalg.transpose
      ins(%input : tensor<1x63xi8>)
      outs(%input_empty : tensor<63x1xi8>) permutation = [1, 0]
  %zero = arith.constant dense<0> : tensor<2x1xi32>
  %core = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %normalized_input : tensor<2x63xi8>, tensor<63x1xi8>)
      outs(%zero : tensor<2x1xi32>) -> tensor<2x1xi32>
  %output_empty = tensor.empty() : tensor<1x2xi32>
  %restored = linalg.transpose
      ins(%core : tensor<2x1xi32>)
      outs(%output_empty : tensor<1x2xi32>) permutation = [1, 0]
  %bias = arith.constant dense<[-128, 127]> : tensor<2xi32>
  %bias_empty = tensor.empty() : tensor<1x2xi32>
  %broadcast = linalg.broadcast
      ins(%bias : tensor<2xi32>)
      outs(%bias_empty : tensor<1x2xi32>) dimensions = [0]
  %result = linalg.add
      ins(%restored, %broadcast : tensor<1x2xi32>, tensor<1x2xi32>)
      outs(%bias_empty : tensor<1x2xi32>) -> tensor<1x2xi32>
  return %result : tensor<1x2xi32>
}

// DEFAULT-LABEL: func.func @linear_exact
// DEFAULT: arith.constant dense<1> : tensor<1x1xi8>
// DEFAULT: arith.constant dense<{{.*}}> : tensor<1x65xi8>
// DEFAULT: tensor.extract_slice {{.*}}[0, 0] [63, 1] [1, 1]
// DEFAULT: tensor.extract_slice {{.*}}[63, 0] [1, 1] [1, 1]
// DEFAULT: tensor.concat dim(0) {{.*}} -> tensor<65x1xi8>
// DEFAULT: linalg.matmul {cim.onnx.matmul_integer} ins({{.*}}tensor<1x65xi8>, tensor<65x1xi8>)
// DEFAULT-NOT: linalg.broadcast
// DEFAULT-NOT: linalg.add
// DEFAULT: return
func.func @linear_exact(%input: tensor<1x64xi8>) -> tensor<1x1xi32> {
  %weight = arith.constant dense<0> : tensor<1x64xi8>
  %input_empty = tensor.empty() : tensor<64x1xi8>
  %normalized_input = linalg.transpose
      ins(%input : tensor<1x64xi8>)
      outs(%input_empty : tensor<64x1xi8>) permutation = [1, 0]
  %zero = arith.constant dense<0> : tensor<1x1xi32>
  %core = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %normalized_input : tensor<1x64xi8>, tensor<64x1xi8>)
      outs(%zero : tensor<1x1xi32>) -> tensor<1x1xi32>
  %output_empty = tensor.empty() : tensor<1x1xi32>
  %restored = linalg.transpose
      ins(%core : tensor<1x1xi32>)
      outs(%output_empty : tensor<1x1xi32>) permutation = [1, 0]
  %bias = arith.constant dense<7> : tensor<1xi32>
  %bias_empty = tensor.empty() : tensor<1x1xi32>
  %broadcast = linalg.broadcast
      ins(%bias : tensor<1xi32>)
      outs(%bias_empty : tensor<1x1xi32>) dimensions = [0]
  %result = linalg.add
      ins(%restored, %broadcast : tensor<1x1xi32>, tensor<1x1xi32>)
      outs(%bias_empty : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %result : tensor<1x1xi32>
}

// DEFAULT-LABEL: func.func @conv_tail
// DEFAULT: arith.constant dense<{{.*}}> : tensor<1x3xi8>
// DEFAULT: arith.constant dense<1> : tensor<1x4xi8>
// DEFAULT: tensor.concat dim(0) {{.*}} -> tensor<3x4xi8>
// DEFAULT: linalg.matmul {cim.onnx.matmul_integer} ins({{.*}}tensor<1x3xi8>, tensor<3x4xi8>)
// DEFAULT-NOT: linalg.broadcast
// DEFAULT-NOT: linalg.add
// DEFAULT: return
func.func @conv_tail(%input: tensor<2x4xi8>) -> tensor<1x1x2x2xi32> {
  %weight = arith.constant dense<0> : tensor<1x2xi8>
  %zero = arith.constant dense<0> : tensor<1x4xi32>
  %core = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<1x2xi8>, tensor<2x4xi8>)
      outs(%zero : tensor<1x4xi32>) -> tensor<1x4xi32>
  %inner = tensor.expand_shape %core [[0, 1], [2]]
      output_shape [1, 1, 4] : tensor<1x4xi32> into tensor<1x1x4xi32>
  %outer = tensor.expand_shape %inner [[0], [1], [2, 3]]
      output_shape [1, 1, 2, 2] : tensor<1x1x4xi32> into tensor<1x1x2x2xi32>
  %bias = arith.constant dense<-3> : tensor<1xi32>
  %bias_empty = tensor.empty() : tensor<1x1x2x2xi32>
  %broadcast = linalg.broadcast
      ins(%bias : tensor<1xi32>)
      outs(%bias_empty : tensor<1x1x2x2xi32>) dimensions = [0, 2, 3]
  %result = linalg.add
      ins(%outer, %broadcast : tensor<1x1x2x2xi32>, tensor<1x1x2x2xi32>)
      outs(%bias_empty : tensor<1x1x2x2xi32>) -> tensor<1x1x2x2xi32>
  return %result : tensor<1x1x2x2xi32>
}

// DEFAULT-LABEL: func.func @linear_wide
// DEFAULT: linalg.broadcast
// DEFAULT: linalg.add
// DEFAULT: return
func.func @linear_wide(%input: tensor<1x2xi8>) -> tensor<1x1xi32> {
  %weight = arith.constant dense<0> : tensor<1x2xi8>
  %input_empty = tensor.empty() : tensor<2x1xi8>
  %normalized_input = linalg.transpose
      ins(%input : tensor<1x2xi8>)
      outs(%input_empty : tensor<2x1xi8>) permutation = [1, 0]
  %zero = arith.constant dense<0> : tensor<1x1xi32>
  %core = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %normalized_input : tensor<1x2xi8>, tensor<2x1xi8>)
      outs(%zero : tensor<1x1xi32>) -> tensor<1x1xi32>
  %output_empty = tensor.empty() : tensor<1x1xi32>
  %restored = linalg.transpose
      ins(%core : tensor<1x1xi32>)
      outs(%output_empty : tensor<1x1xi32>) permutation = [1, 0]
  %bias = arith.constant dense<128> : tensor<1xi32>
  %bias_empty = tensor.empty() : tensor<1x1xi32>
  %broadcast = linalg.broadcast
      ins(%bias : tensor<1xi32>)
      outs(%bias_empty : tensor<1x1xi32>) dimensions = [0]
  %result = linalg.add
      ins(%restored, %broadcast : tensor<1x1xi32>, tensor<1x1xi32>)
      outs(%bias_empty : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %result : tensor<1x1xi32>
}

// NOEXTRA-LABEL: func.func @linear_tail
// NOEXTRA-NOT: linalg.broadcast
// NOEXTRA-NOT: linalg.add
// NOEXTRA: return
// NOEXTRA-LABEL: func.func @linear_exact
// NOEXTRA: linalg.broadcast
// NOEXTRA: linalg.add
// NOEXTRA: return
// NOEXTRA-LABEL: func.func @conv_tail
// NOEXTRA-NOT: linalg.broadcast
// NOEXTRA-NOT: linalg.add
// NOEXTRA: return

// DISABLED-LABEL: func.func @linear_tail
// DISABLED: linalg.broadcast
// DISABLED: linalg.add
// DISABLED: return
// DISABLED-LABEL: func.func @linear_exact
// DISABLED: linalg.broadcast
// DISABLED: linalg.add
// DISABLED: return
// DISABLED-LABEL: func.func @conv_tail
// DISABLED: linalg.broadcast
// DISABLED: linalg.add
// DISABLED: return
