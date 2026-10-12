target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q4rows_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4rows_linear.cu"
)
