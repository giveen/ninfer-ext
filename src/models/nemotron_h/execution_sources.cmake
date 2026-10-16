# Execution side of the Nemotron-H family: the forward that composes the landed ops into the three
# mixers and the block, the Program that owns the family's state -- the Mamba convolution history
# and fp32 recurrent plane per lane, and the attention K/V -- and the attention state with the route
# over it. It is kept apart from the loading manifest for the same reason the other families are --
# immutable loading and mutable execution have separate compile owners.
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/attention_state.cpp")
