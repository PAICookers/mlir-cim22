// RUN: mlir-cim22-opt %s -split-input-file -verify-diagnostics

// Software-only evidence: VMM-N01 through VMM-N09 verifier/parser coverage.

// VMM-N01: input rank.
func.func @input_rank(%input: tensor<1x64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects input shape [64]}}
  %0 = cim.vmm %input, %weight : tensor<1x64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N02: input K, including dynamic K through one representative case.
func.func @input_dynamic_k(%input: tensor<?xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects input shape [64]}}
  %0 = cim.vmm %input, %weight : tensor<?xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N03: input dtype.
func.func @input_dtype(%input: tensor<64xi16>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{operand #0 must be ranked tensor of 8-bit signless integer values}}
  %0 = "cim.vmm"(%input, %weight) : (tensor<64xi16>, tensor<16x64xi8>) -> tensor<16xi21>
  return
}

// -----

// VMM-N04: weight rank.
func.func @weight_rank(%input: tensor<64xi8>, %weight: tensor<1024xi8>) {
  // expected-error@+1 {{expects weight shape [16, 64] in [N, K] order}}
  %0 = cim.vmm %input, %weight : tensor<64xi8>, tensor<1024xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N05: transposed weight axes.
func.func @weight_axes(%input: tensor<64xi8>, %weight: tensor<64x16xi8>) {
  // expected-error@+1 {{expects weight shape [16, 64] in [N, K] order}}
  %0 = cim.vmm %input, %weight : tensor<64xi8>, tensor<64x16xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N06: weight dtype.
func.func @weight_dtype(%input: tensor<64xi8>, %weight: tensor<16x64xi16>) {
  // expected-error@+1 {{operand #1 must be ranked tensor of 8-bit signless integer values}}
  %0 = "cim.vmm"(%input, %weight) : (tensor<64xi8>, tensor<16x64xi16>) -> tensor<16xi21>
  return
}

// -----

// VMM-N07: result shape, including static/dynamic/rank errors through one case.
func.func @result_shape(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects result shape [16]}}
  %0 = cim.vmm %input, %weight : tensor<64xi8>, tensor<16x64xi8> -> tensor<?xi21>
  return
}

// -----

// VMM-N08: result dtype.
func.func @result_dtype(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{result #0 must be ranked tensor of 21-bit signless integer values}}
  %0 = "cim.vmm"(%input, %weight) : (tensor<64xi8>, tensor<16x64xi8>) -> tensor<16xi32>
  return
}

// -----

// VMM-N09: arity.
func.func @operand_arity(%input: tensor<64xi8>) {
  // expected-error@+1 {{expected 2 operands, but found 1}}
  %0 = "cim.vmm"(%input) : (tensor<64xi8>) -> tensor<16xi21>
  return
}
