add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input
  ninfer::json)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

# Host-pointer-for-registered-memory diagnostic. It answers whether a kernel can read pinned host
# memory at the address the CPU holds for it, which is the one Windows port question that cannot be
# settled on Linux (docs/maintainer/windows-port.md). It loads no model and needs no artifact, and it
# is the report a Windows user pastes into an issue, so it is an app rather than a test. The source is
# a .cu because it defines and launches a kernel, which only nvcc parses.
add_executable(ninfer-probe probe/main.cu)
ninfer_internal_includes(ninfer-probe)
target_link_libraries(ninfer-probe PRIVATE CUDA::cudart)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp
  perplexity/reference.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)

# Offline EXL3 weight quantizer. It reads a BF16 artifact and writes a converter source store; it
# is a one-shot tool, not a serving product, and never runs inside an Engine Program.
add_executable(ninfer-quantize
  quantize/main.cpp
  quantize/options.cpp
  quantize/parameter_reader.cpp
  quantize/hessian_io.cpp
  quantize/source_writer.cpp
  quantize/trace_reader.cpp
  quantize/calibration.cpp)
ninfer_internal_includes(ninfer-quantize)
target_include_directories(ninfer-quantize PRIVATE ${PROJECT_SOURCE_DIR}/apps/quantize)
target_link_libraries(ninfer-quantize PRIVATE
  ninfer_engine ninfer_quantize ninfer_artifact ninfer::json)

# Offline EXL3 quantization sensitivity measurement: perturbs one projection at a time and scores the
# loss, the input the bit allocation needs. One-shot tool, not a serving product.
add_executable(ninfer-sensitivity
  sensitivity/main.cpp
  quantize/trace_reader.cpp)
ninfer_internal_includes(ninfer-sensitivity)
target_include_directories(ninfer-sensitivity PRIVATE ${PROJECT_SOURCE_DIR}/apps/quantize)
target_link_libraries(ninfer-sensitivity PRIVATE
  ninfer_engine ninfer_artifact ninfer::json)
