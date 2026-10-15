# Execution side of the Nemotron-H family: the forward that composes the landed ops into the three
# mixers and the block. It is kept apart from the loading manifest for the same reason the other
# families are -- immutable loading and mutable execution have separate compile owners -- and the
# program that owns the state joins this manifest when it lands.
target_sources(ninfer_model_runtime PRIVATE "${CMAKE_CURRENT_LIST_DIR}/forward.cpp")
