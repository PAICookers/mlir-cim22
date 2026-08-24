//===- CIMFrameOps.td - CIMFrame operations --------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEOPS
#define CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEOPS

include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.td"

class CIMFrame_RoutedOp<string mnemonic, list<Trait> traits = []>
    : CIMFrame_Op<mnemonic, !listconcat(traits, [HasParent<"ModuleOp">])> {
  let arguments = (ins DenseI32ArrayAttr:$route);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def CIMFrame_StartInt8OnceOp
    : CIMFrame_Op<"start_int8_once", [HasParent<"ModuleOp">]> {
  let summary = "Request an INT8 start-once control prefix";
  let description = [{
    An effectful packet-stage command that lowers to one INT8 control packet
    followed by one once-work packet. It is not a complete CIM execution plan and has
    no hardware artifact translation.

    The route remains supplied manually at this M2 boundary; M4
    `map-cim-schedule` owns production route calculation and verification.
    Project policy emits complete control for every command and never relies on
    reset or retained state. Runtime owns once issuance after semantic setup.
  }];
  let arguments = (ins DenseI32ArrayAttr:$route, I32Attr:$macro);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def CIMFrame_WriteInt8WeightsOp
    : CIMFrame_Op<"write_int8_weights", [HasParent<"ModuleOp">]> {
  let summary = "Submit one laid-out INT8 weight tile";
  let description = [{
    An effectful command carrying 256 opaque i32 words in semantic CIM address
    order. It does not define frame payload encoding or a deployable artifact.

    The route remains supplied manually at this M2 boundary; M4
    `map-cim-schedule` owns production route calculation and verification.
    Project policy emits complete control for every command and never relies on
    reset or retained state. Runtime owns once issuance after semantic setup.
    The accepted logical weight-to-word mapping is owned by the layout helper.
    FIXME(CTQ-020): Supplier fixture agreement is not board verification.
  }];
  let arguments = (ins DenseI32ArrayAttr:$route, I32Attr:$macro,
      AnyI32ElementsAttr:$words);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def CIMFrame_ControlInt8PacketOp
    : CIMFrame_Op<"control_int8_packet", [HasParent<"ModuleOp">]> {
  let summary = "Represent one typed INT8 control packet";
  let description = [{
    A semantic single-flit control packet corresponding provisionally to type
    `1000`, INT8 mode, and the selected Macro. It does not contain raw bits.
  }];
  let arguments = (ins DenseI32ArrayAttr:$route, I32Attr:$macro);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def CIMFrame_WorkOncePacketOp : CIMFrame_RoutedOp<"work_once_packet"> {
  let summary = "Represent one typed once-work packet";
  let description = [{
    A semantic single-flit work packet corresponding provisionally to type
    `1001` with `NN_RUN_io=01`. It does not contain raw bits.
  }];
}

def CIMFrame_CIMInt8WeightPacketOp
    : CIMFrame_Op<"cim_int8_weight_packet", [HasParent<"ModuleOp">]> {
  let summary = "Represent one typed compound CIM INT8 weight packet";
  let description = [{
    A complete semantic CIM configuration packet carrying 256 opaque i32 words
    in address order. It has no raw type, length, address, body, or mode fields.

    The first canonical profile represents exactly one 16x64 weight tile as
    256 words in semantic address order. Raw emission must validate the
    corresponding semantic configuration-scope address coverage before serialization.
    HWSRC-044 preserves words as opaque two's-complement-derived bit patterns;
    raw emission copies each word bit-for-bit into CIM_data_in.
    The accepted logical weight-to-word mapping is owned by the layout helper.
    FIXME(CTQ-020): Supplier fixture agreement is not board verification.
  }];
  let arguments = (ins DenseI32ArrayAttr:$route,
      AnyI32ElementsAttr:$words);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def CIMFrame_WriteInputCacheInt8PacketOp
    : CIMFrame_Op<"write_input_cache_int8_packet", [HasParent<"ModuleOp">]> {
  let summary = "Represent one INT8 input Cache configuration packet";
  let description = [{
    A complete semantic Cache configuration packet for one input Cache row.
    The canonical INT8 profile uses 16 64-bit body flits for rows 0 through 7.
    The operation contains semantic words only; transport byte order is not
    part of this operation.
  }];
  let arguments = (ins DenseI32ArrayAttr:$route, I32Attr:$macro,
      I32Attr:$cache_row, DenseI64ArrayAttr:$words);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def CIMFrame_ConfigureTestReturnRoutePacketOp
    : CIMFrame_RoutedOp<"configure_test_return_route_packet"> {
  let summary = "Represent one single-flit test return-route configuration";
  let description = [{
    A single-flit `1010` test return-route configuration. The three return
    distances describe the test host relative to the target core.
  }];
  let arguments = (ins DenseI32ArrayAttr:$route, DenseI32ArrayAttr:$test_core);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def CIMFrame_ReadOutputCacheInt8PacketOp
    : CIMFrame_RoutedOp<"read_output_cache_int8_packet"> {
  let summary = "Represent one single-flit INT8 output Cache read request";
  let description = [{
    A single-flit `0101` output Cache read request. The Macro is selected by
    the immediately preceding control packet and is intentionally absent from
    this operation.
  }];
  let arguments = (ins DenseI32ArrayAttr:$route, I32Attr:$cache_address);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEOPS
