// Binary storage for existing transaction fields; no arithmetic or frame codec.
#include "ModelData.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/EndianStream.h"
#include <type_traits>

using namespace cim22::execution;
namespace cim22::runtime {
namespace {
// Fields use little-endian 64-bit integers, length-prefixed arrays/strings.
// Version changes are required for layout or numerical contract changes.
// Version 2 uses candidate BF16 prealigned weights; version 1 used supplier
// arithmetic. Reject old artifacts rather than silently changing their meaning.
// Nothing stores pointers, C++ object layouts, JSON, or source MLIR in the
// payload.
template <typename IO, typename T>
void work(IO &io, T &v) {
  io(v.workId, v.coreSlot, v.macroSlot, v.route);
}
template <typename IO, typename T>
void group(IO &io, T &v) {
  io(v.groupId, v.works);
}
template <typename IO, typename T>
void weight(IO &io, T &v) {
  io(v.groupId, v.workId, v.coreSlot, v.macroSlot, v.resource, v.route, v.words,
     v.values, v.exponents, v.dataType);
}
template <typename IO, typename T>
void packet(IO &io, T &v) {
  io(v.kind, v.route, v.groupId, v.workId, v.macroSlot, v.cacheAddress,
     v.testCore, v.dataType);
}
template <typename IO, typename T>
void input(IO &io, T &v) {
  io(v.groupId, v.workId, v.macroSlot, v.inputSlot, v.dataType);
}
template <typename IO, typename T>
void readback(IO &io, T &v) {
  io(v.groupId, v.workId, v.macroSlot, v.outputCacheAddress, v.route,
     v.testCore, v.dataType);
}

struct Writer {
  llvm::raw_ostream &stream;
  template <typename... T>
  void operator()(const T &...v) {
    (value(v), ...);
  }
  template <typename T>
  void value(const T &v) {
    if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
      llvm::support::endian::write<uint64_t>(stream, static_cast<uint64_t>(v),
                                             llvm::endianness::little);
    } else {
      value(static_cast<uint64_t>(v.size()));
      for (auto &item : v)
        value(item);
    }
  }
  void value(llvm::StringRef v) {
    value(static_cast<uint64_t>(v.size()));
    stream << v;
  }
  void value(const std::string &v) { value(llvm::StringRef(v)); }
  void value(const CIMWork &v) { work(*this, v); }
  void value(const CIMGroup &v) { group(*this, v); }
  void value(const StaticWeightSection &v) { weight(*this, v); }
  void value(const CIMFramePacket &v) { packet(*this, v); }
  void value(const DynamicInputBinding &v) { input(*this, v); }
  void value(const ReadbackBinding &v) { readback(*this, v); }
};

struct Reader {
  llvm::DataExtractor data;
  llvm::DataExtractor::Cursor cursor{0};
  bool invalid = false;
  explicit Reader(llvm::StringRef bytes) : data(bytes, true, 8) {}
  template <typename... T>
  void operator()(T &...v) {
    (value(v), ...);
  }
  uint64_t count() {
    uint64_t n = data.getU64(cursor);
    if (n > data.size() - cursor.tell()) {
      invalid = true;
      return 0;
    }
    return n;
  }
  template <typename T>
  void value(T &v) {
    if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
      uint64_t n = data.getU64(cursor);
      v = static_cast<T>(n);
      invalid |= static_cast<uint64_t>(v) != n;
      if constexpr (std::is_same_v<T, CIMDataType>)
        invalid |= n > static_cast<uint64_t>(CIMDataType::BF16);
      if constexpr (std::is_same_v<T, CIMPacketKind>)
        invalid |= n > static_cast<uint64_t>(CIMPacketKind::OutputCacheRead);
    } else {
      uint64_t n = count();
      if (n > (data.size() - cursor.tell()) / 8) {
        invalid = true;
        return;
      }
      v.resize(n);
      for (auto &item : v)
        value(item);
    }
  }
  template <typename T, size_t N>
  void value(std::array<T, N> &v) {
    if (count() != N) {
      invalid = true;
      return;
    }
    for (auto &item : v)
      value(item);
  }
  void value(std::string &v) {
    uint64_t size = count();
    v = data.getBytes(cursor, size).str();
  }
  void value(CIMWork &v) { work(*this, v); }
  void value(CIMGroup &v) { group(*this, v); }
  void value(StaticWeightSection &v) { weight(*this, v); }
  void value(CIMFramePacket &v) { packet(*this, v); }
  void value(DynamicInputBinding &v) { input(*this, v); }
  void value(ReadbackBinding &v) { readback(*this, v); }
};
} // namespace

std::string encodeModelData(const ModelData &s) {
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  Writer out{stream};
  out(uint64_t{2}, s.numInputs, s.shapes, s.kernelNames, s.outputSchema,
      static_cast<uint64_t>(s.transactions.size()));
  for (auto &t : s.transactions)
    out(t.getTargetProfile(), t.getTargetProfileVersion(),
        t.getExecutionPlanVersion(), t.getGroups(), t.getStaticWeights(),
        t.getPackets(), t.getDynamicInputs(), t.getReadbacks(), t.getFlits());
  return bytes;
}

llvm::Error decodeModelData(llvm::StringRef bytes, ModelData &s) {
  Reader in(bytes);
  uint64_t version = 0;
  in(version);
  if (auto error = in.cursor.takeError())
    return error;
  if (version != 2)
    return llvm::createStringError(
        "unsupported CIM22 binary model data version; recompile the model");
  in(s.numInputs, s.shapes, s.kernelNames, s.outputSchema);
  uint64_t count = in.count();
  for (uint64_t i = 0; i < count && in.cursor && !in.invalid; ++i) {
    std::string profile;
    int64_t profileVersion = 0, planVersion = 0;
    std::vector<CIMGroup> groups;
    std::vector<StaticWeightSection> weights;
    std::vector<CIMFramePacket> packets;
    std::vector<DynamicInputBinding> inputs;
    std::vector<ReadbackBinding> readbacks;
    std::vector<uint64_t> flits;
    in(profile, profileVersion, planVersion, groups, weights, packets, inputs,
       readbacks, flits);
    s.transactions.emplace_back(std::move(profile), profileVersion, planVersion,
                                std::move(groups), std::move(weights),
                                std::move(packets), std::move(inputs),
                                std::move(readbacks), std::move(flits));
  }
  if (auto error = in.cursor.takeError())
    return error;
  if (in.invalid || in.cursor.tell() != bytes.size() || s.numInputs < 0 ||
      static_cast<size_t>(s.numInputs) > s.shapes.size() ||
      s.kernelNames.size() != s.transactions.size())
    return llvm::createStringError("invalid CIM22 binary model data");
  for (auto &shape : s.shapes)
    for (int64_t dim : shape)
      if (dim <= 0)
        return llvm::createStringError("invalid CIM22 static tensor shape");
  return llvm::Error::success();
}
} // namespace cim22::runtime
