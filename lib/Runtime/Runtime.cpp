// Execution-only runtime: no MLIR parser, passes, or LLVM JIT.
#include "CIM22/Execution/Runtime.h"
#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Execution/ModelABI.h"
#include "CIM22/Runtime/CIMBoardRunner.h"
#include "CIM22/Runtime/UsbCIMDevice.h"
#include "Session.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstring>
#ifdef __linux__
#include <dlfcn.h>
#endif
using cim22::runtime::Session;
using llvm::SmallVector;
namespace {
thread_local Session *activeSession = nullptr;
}

void cim22::runtime::copyError(llvm::StringRef message, char *buffer,
                               size_t capacity) {
  if (buffer && capacity) {
    size_t count = std::min(capacity - 1, message.size());
    std::memcpy(buffer, message.data(), count);
    buffer[count] = '\0';
  }
}

extern "C" void *cim22_session_load_shared(const char *path, char *error,
                                           size_t capacity) {
  auto session = std::make_unique<Session>();
#ifdef __linux__
  void *library = path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : nullptr;
  if (!library) {
    cim22::runtime::copyError(path ? dlerror() : "missing model library path",
                              error, capacity);
    return nullptr;
  }
  session->code = std::shared_ptr<void>(library, [](void *p) { dlclose(p); });
  auto getModel = reinterpret_cast<const CIM22ModelV1 *(*)()>(
      dlsym(library, "cim22_model_v1"));
#elif defined(_WIN32)
  using llvm::sys::DynamicLibrary;
  std::string message;
  auto library =
      path ? DynamicLibrary::getLibrary(path, &message) : DynamicLibrary();
  if (!library.isValid()) {
    cim22::runtime::copyError(path ? message : "missing model library path",
                              error, capacity);
    return nullptr;
  }
  // Non-permanent loading lets Session release the DLL before deleting it.
  session->code = std::shared_ptr<DynamicLibrary>(
      new DynamicLibrary(library), [](DynamicLibrary *p) {
        DynamicLibrary::closeLibrary(*p);
        delete p;
      });
  auto getModel = reinterpret_cast<const CIM22ModelV1 *(*)()>(
      library.getAddressOfSymbol("cim22_model_v1"));
#else
#error Unsupported native model platform
#endif
  const CIM22ModelV1 *model = getModel ? getModel() : nullptr;
  if (!model || model->abi_version != 1 || !model->data || !model->invoke) {
    cim22::runtime::copyError("incompatible CIM22 model ABI", error, capacity);
    return nullptr;
  }
  if (auto failure = cim22::runtime::decodeModelData(
          {reinterpret_cast<const char *>(model->data), model->data_size},
          session->model)) {
    cim22::runtime::copyError(llvm::toString(std::move(failure)), error,
                              capacity);
    return nullptr;
  }
  if (session->model.backend == cim22::runtime::RuntimeBackend::Software)
    session->runner = std::make_unique<::cim22::execution::CIMSoftwareRunner>();
  session->invoke = model->invoke;
  return session.release();
}

extern "C" void
_mlir_ciface_cim22_dispatch(int64_t id, StridedMemRefType<uint16_t, 2> *input,
                            StridedMemRefType<uint16_t, 2> *output) {
  Session &session = *activeSession;
  std::fill_n(output->data + output->offset,
              output->sizes[0] * output->sizes[1], uint16_t{0});
  if (!session.error.empty())
    return;
  if (id < 0 || static_cast<size_t>(id) >= session.model.transactions.size() ||
      input->sizes[1] != 64 || input->strides[1] != 1 ||
      input->strides[0] != 64 || output->sizes[1] != 16 ||
      output->strides[1] != 1 || output->strides[0] != 16) {
    session.error = "invalid native CIM dispatch descriptor";
    return;
  }
  SmallVector<::cim22::execution::CIMInputView> inputs;
  SmallVector<::cim22::execution::CIMOutputView> outputs;
  for (int64_t i = 0; i < input->sizes[0]; ++i)
    inputs.push_back({{}, {input->data + input->offset + i * 64, 64}});
  for (int64_t i = 0; i < output->sizes[0]; ++i)
    outputs.push_back({{}, {output->data + output->offset + i * 16, 16}});
  if (!session.runner) {
    session.error =
        session.model.backend == cim22::runtime::RuntimeBackend::USB
            ? "USB runtime is not configured; call configure_usb first"
            : "runtime backend is unavailable";
    return;
  }
  ::cim22::execution::CIMRunOutputs runOutputs{outputs};
  ++session.dispatches;
  if (auto error = session.runner->run(session.model.transactions[id], {inputs},
                                       runOutputs))
    session.error =
        session.model.kernelNames[id] + ": " + llvm::toString(std::move(error));
}

extern "C" void cim22_session_destroy(void *handle) {
  delete static_cast<Session *>(handle);
}
extern "C" int64_t cim22_session_num_inputs(void *handle) {
  return handle ? static_cast<Session *>(handle)->model.numInputs : -1;
}
extern "C" int64_t cim22_session_num_outputs(void *handle) {
  return handle ? static_cast<Session *>(handle)->model.shapes.size() -
                      static_cast<Session *>(handle)->model.numInputs
                : -1;
}
extern "C" int64_t cim22_session_rank(void *handle, int64_t tensor) {
  auto *s = static_cast<Session *>(handle);
  return s && tensor >= 0 &&
                 static_cast<size_t>(tensor) < s->model.shapes.size()
             ? s->model.shapes[tensor].size()
             : -1;
}
extern "C" int64_t cim22_session_dim(void *handle, int64_t tensor,
                                     int64_t axis) {
  auto *s = static_cast<Session *>(handle);
  return axis >= 0 && axis < cim22_session_rank(handle, tensor)
             ? s->model.shapes[tensor][axis]
             : -1;
}
extern "C" int64_t cim22_session_num_transactions(void *handle) {
  return handle ? static_cast<Session *>(handle)->model.transactions.size()
                : -1;
}
extern "C" uint64_t cim22_session_dispatch_count(void *handle) {
  return handle ? static_cast<Session *>(handle)->dispatches : 0;
}
extern "C" const char *cim22_session_error(void *handle) {
  return handle ? static_cast<Session *>(handle)->error.c_str()
                : "null session";
}
extern "C" int cim22_session_invoke(void *handle, void **buffers,
                                    const int64_t *counts, int64_t size) {
  if (!handle)
    return 1;
  auto &s = *static_cast<Session *>(handle);
  std::lock_guard<std::mutex> lock(s.mutex);
  s.error.clear();
  s.dispatches = 0;
  if (!buffers || !counts ||
      size != static_cast<int64_t>(s.model.shapes.size())) {
    s.error = "runtime buffer count mismatch";
    return 1;
  }
  // MLIR ranked descriptor layout: allocated/aligned pointers, offset, then
  // rank sizes and rank strides. memcpy avoids type-punning descriptor words.
  static_assert(sizeof(void *) == sizeof(int64_t), "64-bit runtime required");
  std::vector<std::vector<int64_t>> descriptors(size);
  std::vector<void *> pointers(size), packed(size);
  for (int64_t i = 0; i < size; ++i) {
    auto &shape = s.model.shapes[i];
    auto &desc = descriptors[i];
    desc.resize(3 + 2 * shape.size());
    std::memcpy(desc.data(), &buffers[i], sizeof(void *));
    std::memcpy(desc.data() + 1, &buffers[i], sizeof(void *));
    int64_t elements = 1;
    for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0;
         --dim) {
      desc[3 + dim] = shape[dim];
      desc[3 + shape.size() + dim] = elements;
      if (llvm::MulOverflow(elements, shape[dim], elements)) {
        s.error = "runtime shape element count overflows";
        return 1;
      }
    }
    if (!buffers[i] || counts[i] != elements) {
      s.error = "runtime buffer element count mismatch";
      return 1;
    }
    pointers[i] = desc.data();
    packed[i] = &pointers[i];
  }
  Session *previous = activeSession;
  activeSession = &s;
  s.invoke(packed.data());
  activeSession = previous;
  return s.error.empty() ? 0 : 1;
}

extern "C" int cim22_session_configure_usb(void *handle,
                                           const CIM22UsbConfig *config,
                                           char *error, size_t capacity) {
  if (!handle || !config) {
    cim22::runtime::copyError("invalid USB runtime configuration", error,
                              capacity);
    return 1;
  }
  auto &session = *static_cast<Session *>(handle);
  std::lock_guard<std::mutex> lock(session.mutex);
  if (session.model.backend != cim22::runtime::RuntimeBackend::USB) {
    cim22::runtime::copyError("model artifact backend is not USB", error,
                              capacity);
    return 1;
  }
  if (config->word_byte_order > 1) {
    cim22::runtime::copyError("invalid USB word byte order", error, capacity);
    return 1;
  }
  // Stop the receive loop before checking so late errors cannot be discarded.
  if (session.device)
    session.device->close();
  if (session.device && session.device->isFaulted()) {
    session.runner.reset();
    session.device.reset();
    cim22::runtime::copyError(
        "USB receive/transaction failed before reconfiguration; "
        "device closed, configure_usb may be retried explicitly",
        error, capacity);
    return 1;
  }
#ifdef CIM22_HAS_USB
  cim22::runtime::UsbCIMDeviceConfig deviceConfig;
  deviceConfig.vendorId = config->vendor_id;
  deviceConfig.productId = config->product_id;
  if (config->bus_number != 0xff)
    deviceConfig.busNumber = config->bus_number;
  if (config->device_address != 0xff)
    deviceConfig.deviceAddress = config->device_address;
  deviceConfig.interfaceNumber = config->interface_number;
  deviceConfig.alternateSetting = config->alternate_setting;
  deviceConfig.bulkInEndpoint = config->bulk_in_endpoint;
  deviceConfig.bulkOutEndpoint = config->bulk_out_endpoint;
  deviceConfig.physicalLink = config->physical_link;
  deviceConfig.logicalChannel = config->logical_channel;
  deviceConfig.wordByteOrder =
      config->word_byte_order == 1
          ? cim22::runtime::UsbWordByteOrder::LittleEndian
          : cim22::runtime::UsbWordByteOrder::BigEndian;
  deviceConfig.transferTimeoutMs = config->transfer_timeout_ms;
  deviceConfig.receivePollTimeoutMs = config->receive_poll_timeout_ms;
  deviceConfig.responseTimeoutMs = config->response_timeout_ms;
  // Release the interface before replacement. A failed open leaves dispatch
  // disabled; a previous transaction's failure was reported above.
  session.runner.reset();
  session.device.reset();
  auto device = cim22::runtime::createUsbCIMDevice(deviceConfig);
  if (auto failure = device->open()) {
    cim22::runtime::copyError(llvm::toString(std::move(failure)), error,
                              capacity);
    return 1;
  }
  session.device = std::move(device);
  session.runner = std::make_unique<cim22::runtime::CIMBoardRunner>(
      *session.device, cim22::runtime::CIMBoardRunPolicy{});
  return 0;
#else
  cim22::runtime::copyError("CIM22 runtime was built without USB support",
                            error, capacity);
  return 1;
#endif
}

extern "C" int64_t cim22_session_num_readbacks(void *handle, int64_t id) {
  auto *session = static_cast<Session *>(handle);
  if (!session || id < 0 ||
      static_cast<size_t>(id) >= session->model.transactions.size())
    return -1;
  return session->model.transactions[id].getReadbacks().size();
}

extern "C" int cim22_session_capture_usb(void *handle, int64_t id,
                                         const uint16_t *inputs,
                                         size_t inputCount, uint64_t *flits,
                                         size_t flitCount) {
  if (!handle)
    return 1;
  auto &session = *static_cast<Session *>(handle);
  std::lock_guard<std::mutex> lock(session.mutex);
  session.error.clear();
  if (session.model.backend != cim22::runtime::RuntimeBackend::USB ||
      !session.device) {
    session.error = "raw capture requires a configured USB model";
    return 1;
  }
  if (id < 0 || static_cast<size_t>(id) >= session.model.transactions.size()) {
    session.error = "invalid capture transaction index";
    return 1;
  }
  const auto &transaction = session.model.transactions[id];
  if (!inputs || !flits ||
      inputCount != transaction.getDynamicInputs().size() * 64 ||
      flitCount != transaction.getReadbacks().size() * 7) {
    session.error = "raw capture buffer element count mismatch";
    return 1;
  }
  SmallVector<::cim22::execution::CIMInputView> views;
  for (auto [index, binding] :
       llvm::enumerate(transaction.getDynamicInputs())) {
    if (binding.dataType != ::cim22::execution::CIMDataType::BF16) {
      session.error = "raw capture ABI requires BF16 input bits";
      return 1;
    }
    views.push_back({{}, {inputs + index * 64, 64}});
  }
  cim22::runtime::CIMBoardRunner runner(*session.device, {});
  auto response = runner.capture(transaction, {views});
  if (!response) {
    session.error = llvm::toString(response.takeError());
    return 1;
  }
  for (const auto &readback : response->readbacks) {
    auto *row = flits + readback.outputIndex * 7;
    row[0] = readback.headerFlit;
    std::copy(readback.bodyFlits.begin(), readback.bodyFlits.end(), row + 1);
  }
  return 0;
}
