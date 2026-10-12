cmake_minimum_required(VERSION 3.18)
project(TensorRT85Inference LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "" CACHE PATH "Extracted TensorRT 8.5.1.7 SDK directory")
if(NOT TENSORRT_ROOT)
  message(FATAL_ERROR "Set -DTENSORRT_ROOT=C:/TensorRT/TensorRT-8.5.1.7")
endif()
find_package(CUDAToolkit REQUIRED)
find_path(TRT_INCLUDE_DIR NvInfer.h PATHS "${TENSORRT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(TRT_NVINFER NAMES nvinfer PATHS "${TENSORRT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
add_executable(trt_infer src/main.cpp)
target_include_directories(trt_infer PRIVATE "${TRT_INCLUDE_DIR}")
target_link_libraries(trt_infer PRIVATE "${TRT_NVINFER}" CUDA::cudart)
if(MSVC)
  target_compile_options(trt_infer PRIVATE /W4 /EHsc)
endif()
# Copy the matching TensorRT runtime DLL next to the executable for standalone use.
if(WIN32)
  add_custom_command(TARGET trt_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${TENSORRT_ROOT}/lib/nvinfer.dll" "$<TARGET_FILE_DIR:trt_infer>"
    COMMENT "Copying TensorRT runtime DLL")
endif()
