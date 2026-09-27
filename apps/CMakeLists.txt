add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
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
  quantize/source_writer.cpp)
ninfer_internal_includes(ninfer-quantize)
target_include_directories(ninfer-quantize PRIVATE ${PROJECT_SOURCE_DIR}/apps/quantize)
target_link_libraries(ninfer-quantize PRIVATE
  ninfer_quantize ninfer_artifact ninfer::json)
