# The routed-expert residency measurement: one real artifact, one pass per configuration, and the
# host-link traffic and wall time each of them costs at the widths a request arrives in. It links
# the execution side because it runs the family's own forward, not a replica of it.
add_executable(ninfer_nemotron_h_expert_residency_bench
  "${CMAKE_CURRENT_LIST_DIR}/expert_residency_bench.cpp")
ninfer_internal_includes(ninfer_nemotron_h_expert_residency_bench)
target_link_libraries(ninfer_nemotron_h_expert_residency_bench PRIVATE
  ninfer_model_runtime ninfer_core)
