// Protobuf storage for static execution fields; no arithmetic or frame codec.
#include "ModelData.h"
#include "cim22_model.pb.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>

using namespace cim22::execution;
namespace cim22::runtime {
namespace pb = ::cim22::artifact::v1;
namespace {
llvm::Error invalid(const llvm::Twine &reason) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid CIM22 Protobuf model: " + reason +
                                     "; recompile the model");
}

pb::DataType encodeType(CIMDataType type) {
  switch (type) {
  case CIMDataType::Int8:
    return pb::INT8;
  case CIMDataType::BF16:
    return pb::BF16;
  }
  return pb::DATA_TYPE_UNSPECIFIED;
}
CIMDataType decodeType(pb::DataType type) {
  switch (type) {
  case pb::INT8:
    return CIMDataType::Int8;
  case pb::BF16:
    return CIMDataType::BF16;
  default:
    llvm_unreachable("validated transaction datatype");
  }
}
pb::PacketKind encodeKind(CIMPacketKind kind) {
  switch (kind) {
  case CIMPacketKind::Control:
    return pb::CONTROL;
  case CIMPacketKind::Weight:
    return pb::WEIGHT;
  case CIMPacketKind::WeightExponent:
    return pb::WEIGHT_EXPONENT;
  case CIMPacketKind::Work:
    return pb::WORK;
  case CIMPacketKind::InputCacheWrite:
    return pb::INPUT_CACHE_WRITE;
  case CIMPacketKind::ReturnRoute:
    return pb::RETURN_ROUTE;
  case CIMPacketKind::OutputCacheRead:
    return pb::OUTPUT_CACHE_READ;
  }
  return pb::PACKET_KIND_UNSPECIFIED;
}
CIMPacketKind decodeKind(pb::PacketKind kind) {
  switch (kind) {
  case pb::CONTROL:
    return CIMPacketKind::Control;
  case pb::WEIGHT:
    return CIMPacketKind::Weight;
  case pb::WEIGHT_EXPONENT:
    return CIMPacketKind::WeightExponent;
  case pb::WORK:
    return CIMPacketKind::Work;
  case pb::INPUT_CACHE_WRITE:
    return CIMPacketKind::InputCacheWrite;
  case pb::RETURN_ROUTE:
    return CIMPacketKind::ReturnRoute;
  case pb::OUTPUT_CACHE_READ:
    return CIMPacketKind::OutputCacheRead;
  default:
    llvm_unreachable("validated packet kind");
  }
}
bool validType(pb::DataType type) {
  return type == pb::INT8 || type == pb::BF16;
}

// Validate presence and fixed arrays before copying into native structs.
// Slot zero is valid, so optional identity fields distinguish it from absence.
template <typename Binding>
bool hasBinding(const Binding &b) {
  return b.has_group_id() && b.has_work_id() && b.has_macro_slot();
}
llvm::Error validateTransaction(const pb::Transaction &t,
                                bool requireUSBFields) {
  if (t.target_profile() != "cim22-4x5-v1" || t.target_profile_version() != 1 ||
      t.execution_plan_version() != 1)
    return invalid("unsupported transaction profile/version");
  using Key = std::pair<int64_t, int64_t>;
  std::set<int64_t> groups;
  std::map<Key, const pb::Work *> works;
  for (const auto &group : t.groups()) {
    if (!group.has_group_id() || group.group_id() < 0 ||
        !groups.insert(group.group_id()).second || group.works().empty())
      return invalid("missing/duplicate group or empty works");
    std::set<std::pair<int64_t, int64_t>> placements;
    for (const auto &work : group.works()) {
      if (!work.has_work_id() || !work.has_core_slot() ||
          !work.has_macro_slot() || work.work_id() < 0 ||
          work.core_slot() < 0 || work.core_slot() >= 20 ||
          work.macro_slot() < 0 || work.macro_slot() >= 2 ||
          work.route_size() != 6 ||
          !works.emplace(Key{group.group_id(), work.work_id()}, &work).second ||
          !placements.emplace(work.core_slot(), work.macro_slot()).second)
        return invalid("missing/duplicate work, placement or route length");
    }
  }
  if (works.empty() || t.flits().empty())
    return invalid("empty transaction works/flits");
  std::map<Key, const pb::StaticWeight *> weights;
  std::map<Key, const pb::InputBinding *> inputs;
  std::map<Key, const pb::ReadbackBinding *> readbacks;
  auto placement = [&](const auto &b) {
    auto found = works.find({b.group_id(), b.work_id()});
    return hasBinding(b) && found != works.end() &&
           b.macro_slot() == found->second->macro_slot() &&
           validType(b.data_type());
  };
  for (const auto &w : t.weights()) {
    if (!placement(w) || !w.has_core_slot() ||
        w.core_slot() != works.at({w.group_id(), w.work_id()})->core_slot() ||
        w.resource().empty() || w.route_size() != 6 || w.words_size() != 256 ||
        w.values().size() != 1024 || w.exponents().size() != 16 ||
        !std::equal(w.route().begin(), w.route().end(),
                    works.at({w.group_id(), w.work_id()})->route().begin()) ||
        !weights.emplace(Key{w.group_id(), w.work_id()}, &w).second)
      return invalid("weight binding, array length or datatype");
  }
  for (const auto &b : t.inputs()) {
    if (!placement(b) || !b.has_input_slot() || b.input_slot() < 0 ||
        (requireUSBFields && !b.has_input_cache_address()) ||
        b.input_cache_address() < 0 || b.input_cache_address() >= 8 ||
        !inputs.emplace(Key{b.group_id(), b.work_id()}, &b).second)
      return invalid("input binding or datatype");
  }
  for (const auto &b : t.readbacks()) {
    if (!placement(b) || !b.has_output_cache_address() ||
        b.output_cache_address() < 0 || b.output_cache_address() >= 8 ||
        b.route_size() != 6 || b.test_core_size() != 3 ||
        !std::equal(b.route().begin(), b.route().end(),
                    works.at({b.group_id(), b.work_id()})->route().begin()) ||
        !readbacks.emplace(Key{b.group_id(), b.work_id()}, &b).second)
      return invalid("readback binding, array length or datatype");
  }
  if (weights.size() != works.size() || inputs.size() != works.size() ||
      readbacks.size() != works.size())
    return invalid("incomplete work bindings");
  for (const auto &[key, weight] : weights)
    if (weight->data_type() != inputs.at(key)->data_type() ||
        weight->data_type() != readbacks.at(key)->data_type())
      return invalid("inconsistent work datatypes");
  std::map<Key, std::set<pb::PacketKind>> packetKinds;
  for (int index = 0; index < t.packets_size(); ++index) {
    const auto &p = t.packets(index);
    if (!pb::PacketKind_IsValid(p.kind()) ||
        p.kind() == pb::PACKET_KIND_UNSPECIFIED || !validType(p.data_type()) ||
        !hasBinding(p) || !p.has_cache_address() || p.route_size() != 6 ||
        p.test_core_size() != 3 || p.group_id() < -1 || p.work_id() < -1 ||
        p.macro_slot() < -1 || p.macro_slot() >= 2 || p.cache_address() < -1 ||
        p.cache_address() >= 8)
      return invalid("packet enum, missing field or array/address");
    if (p.group_id() >= 0 && !groups.count(p.group_id()))
      return invalid("packet references unknown group");
    if (p.work_id() >= 0 && !placement(p))
      return invalid("packet references unknown work/placement");
    if (p.kind() == pb::WORK) {
      if (p.group_id() < 0)
        return invalid("work packet missing group");
    } else if (p.kind() != pb::CONTROL && p.work_id() < 0) {
      return invalid("packet missing work binding");
    }
    if (p.work_id() >= 0 &&
        p.data_type() != weights.at({p.group_id(), p.work_id()})->data_type())
      return invalid("packet datatype differs from work");
    packetKinds[{p.group_id(), p.work_id()}].insert(p.kind());
    if (p.work_id() >= 0) {
      Key key{p.group_id(), p.work_id()};
      const auto &route =
          p.kind() == pb::RETURN_ROUTE || p.kind() == pb::OUTPUT_CACHE_READ
              ? readbacks.at(key)->route()
              : weights.at(key)->route();
      if (p.kind() != pb::CONTROL && p.kind() != pb::WORK &&
          !std::equal(p.route().begin(), p.route().end(), route.begin()))
        return invalid("packet route differs from binding");
      if (p.kind() == pb::RETURN_ROUTE) {
        const auto &testCore = readbacks.at(key)->test_core();
        if (!std::equal(p.test_core().begin(), p.test_core().end(),
                        testCore.begin()))
          return invalid("packet test core differs from readback");
        if (index + 2 >= t.packets_size())
          return invalid("incomplete readback packet sequence");
        for (int offset = 1; offset <= 2; ++offset) {
          const auto &next = t.packets(index + offset);
          if (next.kind() !=
                  (offset == 1 ? pb::CONTROL : pb::OUTPUT_CACHE_READ) ||
              next.group_id() != p.group_id() ||
              next.work_id() != p.work_id() ||
              next.macro_slot() != p.macro_slot() ||
              next.data_type() != p.data_type() || next.route_size() != 6 ||
              !std::equal(next.route().begin(), next.route().end(),
                          route.begin()))
            return invalid("inconsistent readback packet sequence");
        }
      }
    }
    if (p.kind() == pb::OUTPUT_CACHE_READ &&
        p.cache_address() !=
            readbacks.at({p.group_id(), p.work_id()})->output_cache_address())
      return invalid("packet readback address differs from binding");
  }
  // Validate binding references; existing frame/runner code owns flit
  // semantics.
  for (const auto &[key, weight] : weights) {
    const auto &kinds = packetKinds[key];
    for (auto kind : {pb::WEIGHT, pb::INPUT_CACHE_WRITE, pb::RETURN_ROUTE,
                      pb::CONTROL, pb::OUTPUT_CACHE_READ})
      if (!kinds.count(kind))
        return invalid("work binding has no matching packet");
    if (weight->data_type() == pb::BF16 && !kinds.count(pb::WEIGHT_EXPONENT))
      return invalid("BF16 weight binding has no exponent packet");
  }
  return llvm::Error::success();
}

llvm::Error validate(const pb::ModelArtifact &model) {
  // TODO: Enforce data_format_version once the artifact contract stabilizes.
  if (!pb::RuntimeBackend_IsValid(model.runtime_backend()))
    return invalid("unsupported runtime backend");
  for (const auto *tensors : {&model.inputs(), &model.outputs()})
    for (const auto &tensor : *tensors) {
      if (tensor.data_type() != pb::FP32)
        return invalid("boundary requires contiguous CPU FP32 tensors");
      int64_t elements = 1;
      for (int64_t dim : tensor.dimensions())
        if (dim <= 0 || llvm::MulOverflow(elements, dim, elements))
          return invalid("static tensor shape/element count");
    }
  std::set<std::string> names;
  for (const auto &kernel : model.kernels()) {
    if (kernel.kernel_name().empty() ||
        !names.insert(kernel.kernel_name()).second || !kernel.has_transaction())
      return invalid("missing/duplicate kernel name or transaction");
    if (auto error = validateTransaction(kernel.transaction(),
                                         model.runtime_backend() == pb::USB))
      return error;
  }
  return llvm::Error::success();
}

template <typename Native, typename Proto>
void encodeBinding(const Native &from, Proto &to) {
  to.set_group_id(from.groupId);
  to.set_work_id(from.workId);
  to.set_macro_slot(from.macroSlot);
  to.set_data_type(encodeType(from.dataType));
}
template <typename Proto, typename Native>
void decodeBinding(const Proto &from, Native &to) {
  to.groupId = from.group_id();
  to.workId = from.work_id();
  to.macroSlot = from.macro_slot();
  to.dataType = decodeType(from.data_type());
}
template <typename Array, typename Field>
void encodeArray(const Array &from, Field *to) {
  to->Assign(from.begin(), from.end());
}
template <typename Field, typename Array>
void decodeArray(const Field &from, Array &to) {
  std::copy(from.begin(), from.end(), to.begin());
}

void encodeTransaction(const CIMTransaction &from, pb::Transaction &to) {
  to.set_target_profile(from.getTargetProfile().str());
  to.set_target_profile_version(from.getTargetProfileVersion());
  to.set_execution_plan_version(from.getExecutionPlanVersion());
  for (const auto &g : from.getGroups()) {
    auto *group = to.add_groups();
    group->set_group_id(g.groupId);
    for (const auto &w : g.works) {
      auto *work = group->add_works();
      work->set_work_id(w.workId);
      work->set_core_slot(w.coreSlot);
      work->set_macro_slot(w.macroSlot);
      encodeArray(w.route, work->mutable_route());
    }
  }
  for (const auto &w : from.getStaticWeights()) {
    auto *weight = to.add_weights();
    encodeBinding(w, *weight);
    weight->set_core_slot(w.coreSlot);
    weight->set_resource(w.resource);
    encodeArray(w.route, weight->mutable_route());
    encodeArray(w.words, weight->mutable_words());
    weight->set_values(w.values.data(), w.values.size());
    weight->set_exponents(w.exponents.data(), w.exponents.size());
  }
  for (const auto &p : from.getPackets()) {
    auto *packet = to.add_packets();
    encodeBinding(p, *packet);
    packet->set_kind(encodeKind(p.kind));
    packet->set_cache_address(p.cacheAddress);
    encodeArray(p.route, packet->mutable_route());
    encodeArray(p.testCore, packet->mutable_test_core());
  }
  for (const auto &b : from.getDynamicInputs()) {
    auto *input = to.add_inputs();
    encodeBinding(b, *input);
    input->set_input_slot(b.inputSlot);
    input->set_input_cache_address(b.inputCacheAddress);
  }
  for (const auto &b : from.getReadbacks()) {
    auto *readback = to.add_readbacks();
    encodeBinding(b, *readback);
    readback->set_output_cache_address(b.outputCacheAddress);
    encodeArray(b.route, readback->mutable_route());
    encodeArray(b.testCore, readback->mutable_test_core());
  }
  encodeArray(from.getFlits(), to.mutable_flits());
}

CIMTransaction decodeTransaction(const pb::Transaction &from) {
  std::vector<CIMGroup> groups;
  std::vector<StaticWeightSection> weights;
  std::vector<CIMFramePacket> packets;
  std::vector<DynamicInputBinding> inputs;
  std::vector<ReadbackBinding> readbacks;
  for (const auto &g : from.groups()) {
    CIMGroup group{g.group_id(), {}};
    for (const auto &w : g.works()) {
      CIMWork work{w.work_id(), w.core_slot(), w.macro_slot()};
      decodeArray(w.route(), work.route);
      group.works.push_back(work);
    }
    groups.push_back(std::move(group));
  }
  for (const auto &w : from.weights()) {
    StaticWeightSection weight;
    decodeBinding(w, weight);
    weight.coreSlot = w.core_slot();
    weight.resource = w.resource();
    decodeArray(w.route(), weight.route);
    weight.words.assign(w.words().begin(), w.words().end());
    std::memcpy(weight.values.data(), w.values().data(), weight.values.size());
    std::memcpy(weight.exponents.data(), w.exponents().data(),
                weight.exponents.size());
    weights.push_back(std::move(weight));
  }
  for (const auto &p : from.packets()) {
    CIMFramePacket packet{decodeKind(p.kind())};
    decodeBinding(p, packet);
    packet.cacheAddress = p.cache_address();
    decodeArray(p.route(), packet.route);
    decodeArray(p.test_core(), packet.testCore);
    packets.push_back(packet);
  }
  for (const auto &b : from.inputs()) {
    DynamicInputBinding input;
    decodeBinding(b, input);
    input.inputSlot = b.input_slot();
    input.inputCacheAddress = b.input_cache_address();
    inputs.push_back(input);
  }
  for (const auto &b : from.readbacks()) {
    ReadbackBinding readback;
    decodeBinding(b, readback);
    readback.outputCacheAddress = b.output_cache_address();
    decodeArray(b.route(), readback.route);
    decodeArray(b.test_core(), readback.testCore);
    readbacks.push_back(readback);
  }
  return CIMTransaction(from.target_profile(), from.target_profile_version(),
                        from.execution_plan_version(), std::move(groups),
                        std::move(weights), std::move(packets),
                        std::move(inputs), std::move(readbacks),
                        {from.flits().begin(), from.flits().end()});
}
} // namespace

llvm::Expected<std::string> encodeModelData(const ModelData &model) {
  if (model.numInputs < 0 ||
      static_cast<size_t>(model.numInputs) > model.shapes.size() ||
      model.kernelNames.size() != model.transactions.size())
    return invalid("signature/kernel counts");
  pb::ModelArtifact artifact;
  artifact.set_data_format_version(1);
  artifact.set_runtime_backend(
      model.backend == RuntimeBackend::Software ? pb::SOFTWARE : pb::USB);
  for (size_t i = 0; i < model.shapes.size(); ++i) {
    auto *tensor = i < static_cast<size_t>(model.numInputs)
                       ? artifact.add_inputs()
                       : artifact.add_outputs();
    tensor->set_data_type(pb::FP32);
    encodeArray(model.shapes[i], tensor->mutable_dimensions());
  }
  for (size_t i = 0; i < model.transactions.size(); ++i) {
    auto *kernel = artifact.add_kernels();
    kernel->set_kernel_name(model.kernelNames[i]);
    encodeTransaction(model.transactions[i], *kernel->mutable_transaction());
  }
  if (auto error = validate(artifact))
    return error;
  std::string bytes;
  if (!artifact.SerializeToString(&bytes))
    return invalid("model serialization failed");
  return bytes;
}

llvm::Error decodeModelData(llvm::StringRef bytes, ModelData &model) {
  pb::ModelArtifact artifact;
  if (bytes.empty() ||
      bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      !artifact.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())))
    return invalid("malformed data (old custom binary is unsupported)");
  if (auto error = validate(artifact))
    return error;
  ModelData decoded;
  decoded.backend = artifact.runtime_backend() == pb::USB
                        ? RuntimeBackend::USB
                        : RuntimeBackend::Software;
  decoded.numInputs = artifact.inputs_size();
  for (const auto *tensors : {&artifact.inputs(), &artifact.outputs()})
    for (const auto &tensor : *tensors)
      decoded.shapes.emplace_back(tensor.dimensions().begin(),
                                  tensor.dimensions().end());
  for (const auto &kernel : artifact.kernels()) {
    decoded.kernelNames.push_back(kernel.kernel_name());
    decoded.transactions.push_back(decodeTransaction(kernel.transaction()));
  }
  model = std::move(decoded);
  return llvm::Error::success();
}
} // namespace cim22::runtime
