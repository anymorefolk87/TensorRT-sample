#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
  }
};

template <class T> struct TRTDestroy {
  void operator()(T* p) const noexcept { if (p) p->destroy(); }
};
template <class T> using TRTPtr = std::unique_ptr<T, TRTDestroy<T>>;

static void cudaCheck(cudaError_t rc, const char* what) {
  if (rc != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc));
}

struct DeviceBuffer {
  void* ptr = nullptr;
  explicit DeviceBuffer(size_t bytes) { cudaCheck(cudaMalloc(&ptr, bytes), "cudaMalloc"); }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

static std::vector<char> readBytes(const fs::path& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open: " + p.string());
  auto n = f.tellg();
  if (n < 0) throw std::runtime_error("Cannot stat file: " + p.string());
  std::vector<char> data(static_cast<size_t>(n));
  f.seekg(0);
  if (!data.empty()) f.read(data.data(), static_cast<std::streamsize>(data.size()));
  if (!f) throw std::runtime_error("Cannot read: " + p.string());
  return data;
}

static void writeBytes(const fs::path& p, const std::vector<char>& data) {
  std::ofstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot create: " + p.string());
  f.write(data.data(), static_cast<std::streamsize>(data.size()));
  if (!f) throw std::runtime_error("Cannot write: " + p.string());
}

static std::vector<int> parseDims(const std::string& s) {
  std::vector<int> out;
  size_t start = 0;
  while (start < s.size()) {
    auto end = s.find_first_of("xX,", start);
    auto token = s.substr(start, end == std::string::npos ? std::string::npos : end - start);
    int d = std::stoi(token);
    if (d <= 0) throw std::runtime_error("All shape dimensions must be positive");
    out.push_back(d);
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (out.empty() || out.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS))
    throw std::runtime_error("Invalid --shape");
  return out;
}

static size_t elementCount(nvinfer1::Dims dims) {
  if (dims.nbDims < 0) throw std::runtime_error("Invalid tensor rank");
  size_t n = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) throw std::runtime_error("Unresolved dynamic dimension");
    if (dims.d[i] && n > std::numeric_limits<size_t>::max() / static_cast<size_t>(dims.d[i]))
      throw std::runtime_error("Tensor size overflow");
    n *= static_cast<size_t>(dims.d[i]);
  }
  return n;
}

static std::string dimsStr(nvinfer1::Dims d) {
  std::string s;
  for (int i = 0; i < d.nbDims; ++i) s += (i ? "x" : "") + std::to_string(d.d[i]);
  return s;
}

int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 6) {
      std::cerr << "Usage: trt_infer <model.engine> <input.f32.bin> <output_dir> [--shape 1x3x224x224]\n"
                   "Supports one FP32 input, FP32 outputs, explicit-batch engines, optimization profile 0.\n";
      return 2;
    }
    fs::path enginePath = argv[1], inputPath = argv[2], outDir = argv[3];
    const bool hasShape = argc == 6;
    if (hasShape && std::string(argv[4]) != "--shape") throw std::runtime_error("Expected --shape");
    auto shape = hasShape ? parseDims(argv[5]) : std::vector<int>{};

    Logger logger;
    auto serialized = readBytes(enginePath);
    TRTPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) throw std::runtime_error("createInferRuntime failed");
    TRTPtr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(serialized.data(), serialized.size())};
    if (!engine) throw std::runtime_error("Engine deserialization failed (check TensorRT version/GPU/plugins)");
    if (engine->hasImplicitBatchDimension()) throw std::runtime_error("Implicit batch engines unsupported");

    TRTPtr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!context) throw std::runtime_error("Cannot create execution context");
    int profiles = engine->getNbOptimizationProfiles();
    if (profiles < 1) throw std::runtime_error("No optimization profiles");
    int perProfile = engine->getNbBindings() / profiles;
    if (perProfile * profiles != engine->getNbBindings()) throw std::runtime_error("Unexpected binding count");
    int inputIndex = -1;
    for (int i = 0; i < perProfile; ++i) {
      if (engine->bindingIsInput(i)) {
        if (inputIndex >= 0) throw std::runtime_error("Only one input supported");
        inputIndex = i;
      }
      if (engine->getBindingDataType(i) != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error(std::string("Only FP32 I/O supported. Unsupported binding: ") + engine->getBindingName(i));
      if (engine->getBindingFormat(i) != nvinfer1::TensorFormat::kLINEAR)
        throw std::runtime_error(std::string("Only linear tensor format supported: ") + engine->getBindingName(i));
    }
    if (inputIndex < 0) throw std::runtime_error("No input found");

    auto inputDims = engine->getBindingDimensions(inputIndex);
    bool dynamic = false;
    for (int i = 0; i < inputDims.nbDims; ++i) dynamic |= inputDims.d[i] < 0;
    if (dynamic) {
      if (!hasShape || shape.size() != static_cast<size_t>(inputDims.nbDims))
        throw std::runtime_error("Dynamic input requires --shape with exactly matching tensor rank");
      nvinfer1::Dims d = inputDims;
      for (int i = 0; i < d.nbDims; ++i) d.d[i] = shape[static_cast<size_t>(i)];
      if (!context->setBindingDimensions(inputIndex, d))
        throw std::runtime_error("--shape outside optimization profile 0 or invalid");
      if (!context->allInputDimensionsSpecified()) throw std::runtime_error("Input dimensions not fully specified");
    } else if (hasShape) {
      if (shape.size() != static_cast<size_t>(inputDims.nbDims)) throw std::runtime_error("Shape rank mismatch");
      for (int i = 0; i < inputDims.nbDims; ++i)
        if (shape[static_cast<size_t>(i)] != inputDims.d[i]) throw std::runtime_error("Shape mismatch with static engine");
    }

    // All bindings must have addresses, including any inactive optimization profiles.
    std::vector<void*> bindings(static_cast<size_t>(engine->getNbBindings()), nullptr);
    std::vector<std::unique_ptr<DeviceBuffer>> buffers;
    std::vector<std::vector<char>> hostBuffers(static_cast<size_t>(perProfile));
    std::vector<size_t> byteSizes(static_cast<size_t>(perProfile), 0);
    auto input = readBytes(inputPath);
    for (int i = 0; i < perProfile; ++i) {
      auto d = context->getBindingDimensions(i);
      size_t elements = elementCount(d);
      if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) throw std::runtime_error("Bytes overflow");
      size_t bytes = elements * sizeof(float);
      byteSizes[static_cast<size_t>(i)] = bytes;
      if (engine->bindingIsInput(i)) {
        if (input.size() != bytes)
          throw std::runtime_error("Input file size is " + std::to_string(input.size()) +
                                   " but engine expects " + std::to_string(bytes) + " bytes (" + dimsStr(d) + " FP32)");
      } else {
        hostBuffers[static_cast<size_t>(i)].resize(bytes);
      }
      buffers.emplace_back(std::make_unique<DeviceBuffer>(bytes ? bytes : 1));
      bindings[static_cast<size_t>(i)] = buffers.back()->ptr;
      std::cout << (engine->bindingIsInput(i) ? "Input " : "Output ")
                << engine->getBindingName(i) << " [" << dimsStr(d) << "]\n";
    }

    cudaStream_t stream{};
    cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");
    try {
      cudaCheck(cudaMemcpyAsync(bindings[static_cast<size_t>(inputIndex)], input.data(), input.size(),
                                cudaMemcpyHostToDevice, stream), "H2D copy");
      if (!context->enqueueV2(bindings.data(), stream, nullptr)) throw std::runtime_error("enqueueV2 failed");
      for (int i = 0; i < perProfile; ++i) {
        if (!engine->bindingIsInput(i)) {
          auto& host = hostBuffers[static_cast<size_t>(i)];
          cudaCheck(cudaMemcpyAsync(host.data(), bindings[static_cast<size_t>(i)],
                                    byteSizes[static_cast<size_t>(i)], cudaMemcpyDeviceToHost, stream), "D2H copy");
        }
      }
      cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      fs::create_directories(outDir);
      for (int i = 0; i < perProfile; ++i) {
        if (!engine->bindingIsInput(i)) {
          fs::path path = outDir / ("output_" + std::to_string(i) + ".f32.bin");
          writeBytes(path, hostBuffers[static_cast<size_t>(i)]);
          std::cout << "Saved: " << path.string() << "\n";
        }
      }
    } catch (...) {
      cudaStreamDestroy(stream);
      throw;
    }
    cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
