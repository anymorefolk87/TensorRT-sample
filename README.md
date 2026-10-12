# TensorRT 8.5 C++ inference example (Windows x64)

A small **C++17 / Visual Studio 2022 / CMake** example that loads an existing TensorRT 8.5.x serialized engine, performs one GPU inference, and writes raw FP32 outputs.

## Prerequisites (offline machine)

- Windows x64, NVIDIA GPU and compatible NVIDIA driver.
- **Full TensorRT 8.5.1.7 Windows SDK ZIP**, extracted, e.g. `C:\TensorRT\TensorRT-8.5.1.7`. This must contain `include\NvInfer.h`, `lib\nvinfer.lib`, and `lib\nvinfer.dll`. The Python `.whl` alone is **not sufficient to build the C++ example**.
- CUDA toolkit compatible with your TensorRT 8.5 SDK (for example, the SDK's CUDA 11.x variant with matching runtime), installed offline. Its MSVC integration is not needed because the project compiles only `.cpp` code.
- Visual Studio 2022 Desktop development with C++, CMake >= 3.18.
- Any additional runtime DLLs your TensorRT build/engine requires (CUDA runtime, plugins, cuDNN, etc.). `nvinfer.dll` is automatically copied to the EXE output directory during build, but other required DLLs must remain discoverable on PATH.

## Build in PowerShell (no Python needed)

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DTENSORRT_ROOT="C:/TensorRT/TensorRT-8.5.1.7" `
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
cmake --build build --config Release
```

Executable: `build\Release\trt_infer.exe`. The build process requires no internet connection once all dependencies are present.

If you want a GUI project, open the folder in Visual Studio 2022 (`File > Open > Folder`), or open the generated `build\TensorRT85Inference.sln`.

## Run

The program accepts a **serialized TensorRT engine** (`.engine`/`.plan`), a raw FP32 input tensor file (contiguous, binary, engine's expected memory layout), and an output folder.

```powershell
.\build\Release\trt_infer.exe model.engine input.f32.bin outputs
```

For engines with dynamic input dimensions:

```powershell
.\build\Release\trt_infer.exe model.engine input.f32.bin outputs --shape 1x3x224x224
```

Output files are named `output_<binding-index>.f32.bin`, preserving output tensor order as binding indices. The program prints each tensor name and dimensions to stdout.

### Generate a synthetic input file for a 1x3x224x224 FP32 model

```powershell
py -3.10 -c "import struct; open('input.f32.bin','wb').write(struct.pack('<' + 'f'*(1*3*224*224), *([0.0]*(1*3*224*224))))"
```

This is a **smoke test**, not accurate model preprocessing. For real inference you must apply your model's normalization, channel order, scaling and layout before writing the raw input file. Example input size: `1*3*224*224*4 = 602112` bytes.

### Creating an engine from ONNX (outside this example)

With TensorRT's `trtexec.exe` available:

```powershell
trtexec.exe --onnx=model.onnx --saveEngine=model.engine
```

For dynamic ONNX models, supply appropriate profile shape options when building the engine (e.g. `--minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:4x3x224x224`, using the actual input name).

**TensorRT engines are generally specific to the TensorRT version, hardware and build settings**. Build or obtain one that works on the target GPU.

## Limitations

- **Exactly one FP32 input**, with **FP32 outputs**, standard contiguous `kLINEAR` tensor formats. It does not perform image decode, resize, channel permutation, float16 handling or postprocessing.
- Explicit-batch TensorRT 8.5 engine only. For dynamic inputs pass `--shape` inside profile 0's range.
- Uses optimization profile 0; does not set shape tensor bindings or support data-dependent output dimensions.
- The example loads a prebuilt engine; it is not an ONNX parser/build tool.

For engines with multiple inputs, FP16/INT8 I/O, shape tensors, dynamic output sizes, or image inputs, the code must be extended.

## Troubleshooting

- **NvInfer.h not found**: check `TENSORRT_ROOT` points at the full Windows SDK, not the Python wheel.
- **nvinfer.lib not found**: confirm `lib` exists in the TensorRT SDK archive.
- **DLL missing**: put CUDA runtime DLL folders and TensorRT's `lib` folder on PATH; plugins and cuDNN DLLs may be needed.
- **Engine deserialization failed**: check TensorRT version, matching NVIDIA GPU, required custom plugins, and engine format.
- **Input file size mismatch**: save raw float32 data exactly matching the input shape and tensor order.

The project does **not** install anything and does not need Python to build or run.
