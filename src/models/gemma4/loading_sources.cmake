target_sources(ninfer_model_loading PRIVATE "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
                                            "${CMAKE_CURRENT_LIST_DIR}/load.cpp")
target_sources(ninfer_model_loading PRIVATE "${CMAKE_CURRENT_LIST_DIR}/forward.cpp")

target_sources(ninfer_model_loading PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cache.cpp")
