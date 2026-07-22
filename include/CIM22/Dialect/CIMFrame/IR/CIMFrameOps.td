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
    An effectful execution-plan command that lowers to one INT8 control packet
    followed by one once-work packet. It is not a complete invocation and has
    no hardware artifact translation.

    TODO(CTQ-003): The route is supplied manually and only field magnitude is
    checked. TODO(CTQ-010): Control is immediately followed by once-work.
    TODO(CTQ-023): Control is resent for every command.
  }];
  let arguments = (ins DenseI32ArrayAttr:$route, I32Attr:$macro);
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

#endif // CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEOPS
