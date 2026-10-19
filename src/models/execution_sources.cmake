# Model-execution pieces that more than one family composes rather than a family's own forward: the
# expert pager is the one owner of how a pool of host-resident routed experts reaches the device,
# and both Qwen4Exp and Nemotron-H drive it. It lives here rather than under either family because
# it is the same code for both -- every per-layer number it needs comes from the pool's own
# `ExpertPoolGeometry`.
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/expert_pager.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/host_expert_lane.cpp")
