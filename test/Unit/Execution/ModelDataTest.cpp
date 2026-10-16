// Native codec round-trip: preserve every transaction field and kernel order.
#include "ModelData.h"
#include <iostream>

using namespace cim22::execution;
using namespace cim22::runtime;

static CIMTransaction transaction(CIMDataType type) {
  CIMWork work{19, 3, 1, {1, 0, 2, 0, 3, 0}};
  StaticWeightSection weight;
  weight.groupId = 7;
  weight.workId = 19;
  weight.coreSlot = 3;
  weight.macroSlot = 1;
  weight.resource = "weight";
  weight.route = work.route;
  weight.dataType = type;
  for (int i = 0; i < 256; ++i)
    weight.words.push_back(i % 2 ? -i : i);
  for (int i = 0; i < 1024; ++i)
    weight.values[i] = static_cast<int8_t>(i);
  for (int i = 0; i < 16; ++i)
    weight.exponents[i] = 120 + i;
  std::vector<CIMFramePacket> packets;
  for (auto kind : {CIMPacketKind::Control, CIMPacketKind::Weight,
                    CIMPacketKind::WeightExponent, CIMPacketKind::Work,
                    CIMPacketKind::InputCacheWrite, CIMPacketKind::ReturnRoute,
                    CIMPacketKind::Control, CIMPacketKind::OutputCacheRead}) {
    CIMFramePacket p{kind};
    p.groupId = 7;
    p.workId = 19;
    p.macroSlot = 1;
    p.route = work.route;
    p.testCore = {1, 2, 3};
    p.cacheAddress = kind == CIMPacketKind::OutputCacheRead ? 7 : -1;
    p.dataType = type;
    packets.push_back(p);
  }
  return CIMTransaction("cim22-4x5-v1", 1, 1, {{7, {work}}}, {weight}, packets,
                        {{7, 19, 1, 47, type, 6}},
                        {{7, 19, 1, 7, work.route, {1, 2, 3}, type}},
                        {0, UINT64_MAX, UINT64_C(0x123456789abcdef0)});
}

template <typename T>
static bool sameBinding(const T &a, const T &b) {
  return a.groupId == b.groupId && a.workId == b.workId &&
         a.macroSlot == b.macroSlot && a.dataType == b.dataType;
}

static bool sameTransaction(const CIMTransaction &a, const CIMTransaction &b) {
  if (a.getTargetProfile() != b.getTargetProfile() ||
      a.getTargetProfileVersion() != b.getTargetProfileVersion() ||
      a.getExecutionPlanVersion() != b.getExecutionPlanVersion() ||
      a.getFlits() != b.getFlits() ||
      a.getGroups().size() != b.getGroups().size() ||
      a.getStaticWeights().size() != b.getStaticWeights().size() ||
      a.getPackets().size() != b.getPackets().size() ||
      a.getDynamicInputs().size() != b.getDynamicInputs().size() ||
      a.getReadbacks().size() != b.getReadbacks().size())
    return false;
  const auto &ga = a.getGroups()[0];
  const auto &gb = b.getGroups()[0];
  if (ga.groupId != gb.groupId || ga.works.size() != gb.works.size())
    return false;
  const auto &wa = ga.works[0];
  const auto &wb = gb.works[0];
  if (wa.workId != wb.workId || wa.coreSlot != wb.coreSlot ||
      wa.macroSlot != wb.macroSlot || wa.route != wb.route)
    return false;
  const auto &sa = a.getStaticWeights()[0];
  const auto &sb = b.getStaticWeights()[0];
  if (!sameBinding(sa, sb) || sa.coreSlot != sb.coreSlot ||
      sa.resource != sb.resource || sa.route != sb.route ||
      sa.words != sb.words || sa.values != sb.values ||
      sa.exponents != sb.exponents)
    return false;
  for (size_t i = 0; i < a.getPackets().size(); ++i) {
    const auto &pa = a.getPackets()[i];
    const auto &pb = b.getPackets()[i];
    if (!sameBinding(pa, pb) || pa.kind != pb.kind || pa.route != pb.route ||
        pa.cacheAddress != pb.cacheAddress || pa.testCore != pb.testCore)
      return false;
  }
  const auto &ia = a.getDynamicInputs()[0];
  const auto &ib = b.getDynamicInputs()[0];
  const auto &ra = a.getReadbacks()[0];
  const auto &rb = b.getReadbacks()[0];
  return sameBinding(ia, ib) && ia.inputSlot == ib.inputSlot &&
         ia.inputCacheAddress == ib.inputCacheAddress && sameBinding(ra, rb) &&
         ra.outputCacheAddress == rb.outputCacheAddress &&
         ra.route == rb.route && ra.testCore == rb.testCore;
}

int main() {
  ModelData model;
  model.numInputs = 1;
  model.backend = RuntimeBackend::Software;
  model.shapes = {{}, {2, 3}}; // Scalar input, ordered tensor output.
  model.kernelNames = {"int8", "bf16"};
  model.transactions = {transaction(CIMDataType::Int8),
                        transaction(CIMDataType::BF16)};
  auto encoded = encodeModelData(model);
  if (!encoded) {
    std::cerr << llvm::toString(encoded.takeError()) << '\n';
    return 1;
  }
  ModelData decoded;
  if (auto error = decodeModelData(*encoded, decoded)) {
    std::cerr << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  // Compare native fields as well as wire bytes, so a field omitted by both
  // encoder and decoder still fails. Check -1 sentinels and signed/raw bits.
  auto reencoded = encodeModelData(decoded);
  if (!reencoded) {
    std::cerr << llvm::toString(reencoded.takeError()) << '\n';
    return 1;
  }
  if (*encoded != *reencoded || decoded.shapes != model.shapes ||
      decoded.kernelNames != model.kernelNames || decoded.numInputs != 1 ||
      decoded.backend != model.backend)
    return 1;
  if (decoded.transactions.size() != model.transactions.size())
    return 1;
  for (size_t i = 0; i < model.transactions.size(); ++i)
    if (!sameTransaction(model.transactions[i], decoded.transactions[i]))
      return 1;
  model.backend = RuntimeBackend::USB;
  auto usb = encodeModelData(model);
  if (!usb) {
    std::cerr << llvm::toString(usb.takeError()) << '\n';
    return 1;
  }
  ModelData usbDecoded;
  if (auto error = decodeModelData(*usb, usbDecoded)) {
    std::cerr << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  if (usbDecoded.backend != RuntimeBackend::USB ||
      !sameTransaction(model.transactions[1], usbDecoded.transactions[1]))
    return 1;
  model.numInputs = static_cast<int64_t>(model.shapes.size()) + 1;
  auto invalidSignature = encodeModelData(model);
  if (invalidSignature)
    return 1;
  llvm::consumeError(invalidSignature.takeError());
  // Failed decoding leaves the previously owned model intact.
  if (auto error = decodeModelData(std::string("\x02", 1), decoded))
    llvm::consumeError(std::move(error));
  else
    return 1;
  auto preserved = encodeModelData(decoded);
  if (!preserved) {
    std::cerr << llvm::toString(preserved.takeError()) << '\n';
    return 1;
  }
  return *preserved == *encoded ? 0 : 1;
}
