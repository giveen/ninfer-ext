ninfer_add_test(ninfer_artifact_reader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reader.cpp"
  LIBRARIES ninfer_artifact)

ninfer_add_test(ninfer_exl3_trellis_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_trellis.cpp"
  LIBRARIES ninfer_artifact)

ninfer_add_test(ninfer_artifact_materialization_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp" "${CMAKE_CURRENT_LIST_DIR}/materialization_cuda_errors.cpp"
  LIBRARIES ninfer_artifact)

# GNU ld and lld only: link.exe has no --wrap, and /alternatename cannot stand in for it (it redirects
# a symbol the linker is looking for and cannot find, while cudart's import library defines these and
# the compiler emits a direct call to the import thunk). On Windows the same translation unit
# interposes the CUDA runtime's import address table at runtime instead, which needs no linker
# support and keeps every assertion.
if(NOT MSVC)
  target_link_options(ninfer_artifact_materialization_test PRIVATE
    "LINKER:--wrap=cudaMalloc"
    "LINKER:--wrap=cudaMallocHost"
    "LINKER:--wrap=cudaFree"
    "LINKER:--wrap=cudaFreeHost"
    "LINKER:--wrap=cudaEventCreateWithFlags"
    "LINKER:--wrap=cudaEventRecord"
    "LINKER:--wrap=cudaMemcpyAsync"
    "LINKER:--wrap=cudaStreamSynchronize")
endif()

add_test(NAME ninfer_artifact_writer_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/writer_interop.py"
    $<TARGET_FILE:ninfer_artifact_materialization_test>)

set_tests_properties(
  ninfer_artifact_writer_interop_test
  PROPERTIES SKIP_RETURN_CODE 77)

set_tests_properties(
  ninfer_artifact_materialization_test
  PROPERTIES SKIP_RETURN_CODE 77)
