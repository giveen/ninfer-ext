ninfer_add_test(ninfer_gemma4_config_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../gemma4/test_config.cpp"
  LIBRARIES ninfer_model_loading)

# Opt-in real-model artifacts, selected by explicit path so a run never depends
# on glob order or modification time. Configure the ones you have and run
# `ctest -L real`; an artifact variable left empty makes its test skip (77).
# The Windows build supplies no artifact.
set(NINFER_ARTIFACT_LOADING "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_loading_real_test")
set(NINFER_ARTIFACT_PREFIX  "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_prefix_real_test")
set(NINFER_ARTIFACT_SCORE   "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_score_real_test")
set(NINFER_ARTIFACT_VISION  "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_vision_workspace_test")
set(NINFER_ARTIFACT_DFLASH2 "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_dflash2_real_test")
set(NINFER_ARTIFACT_MOE     "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_moe_real_test")
set(NINFER_ARTIFACT_GRAMMAR "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_grammar_real_test")
set(NINFER_ARTIFACT_TOOLS "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_tools_real_test")
set(NINFER_ARTIFACT_DFLASH  "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_dflash_real_test")
set(NINFER_ARTIFACT_DFLASH_PREFILL "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_dflash_prefill_real_test")
set(NINFER_ARTIFACT_STREAM  "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_stream_real_test")
set(NINFER_ARTIFACT_RESIDUE "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_spec_residue_real_test")
set(NINFER_ARTIFACT_RESIDUE_EAGLE3 "" CACHE FILEPATH "Artifact for ninfer_qwen3_5_spec_residue_eagle3_real_test")

ninfer_add_real_test(ninfer_qwen3_5_loading_real_test NINFER_ARTIFACT_LOADING
  ARGS --vision --speculative mtp --proposal optimized
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_store.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_context_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_real_test(ninfer_qwen3_5_dflash_prefill_real_test NINFER_ARTIFACT_DFLASH_PREFILL
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_real_test(ninfer_qwen3_5_prefix_real_test NINFER_ARTIFACT_PREFIX
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_real_test(ninfer_qwen3_5_score_real_test NINFER_ARTIFACT_SCORE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_real_test(ninfer_qwen3_5_vision_workspace_test NINFER_ARTIFACT_VISION
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_runtime ninfer_engine)

ninfer_add_real_test(ninfer_qwen3_5_dflash2_real_test NINFER_ARTIFACT_DFLASH2
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_real_test(ninfer_qwen3_5_moe_real_test NINFER_ARTIFACT_MOE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_real_test(ninfer_qwen3_5_dflash_real_test NINFER_ARTIFACT_DFLASH
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../../test_tool_call_parser.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visual_scatter.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_mtp_draft_policy_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_draft_policy.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_suffix_drafter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_suffix_drafter.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_gemma4_tokenizer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_tokenizer.cpp"
  LIBRARIES ninfer_model_runtime ninfer_artifact)
set_tests_properties(ninfer_gemma4_tokenizer_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_gemma4_load_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../gemma4/test_load.cpp"
  LIBRARIES ninfer_model_loading)
set_tests_properties(ninfer_gemma4_load_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_gemma4_model_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../gemma4/test_model.cpp"
  LIBRARIES ninfer_model_loading)
set_tests_properties(ninfer_gemma4_model_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_gemma4_forward_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../gemma4/test_forward.cpp"
  LIBRARIES ninfer_model_loading)
set_tests_properties(ninfer_gemma4_forward_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_gemma4_layer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../gemma4/test_layer.cpp"
  LIBRARIES ninfer_model_loading)
set_tests_properties(ninfer_gemma4_layer_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_ple_gather_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ple_gather.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen3_5_ple_gather_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_real_test(ninfer_qwen3_5_stream_real_test NINFER_ARTIFACT_STREAM
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_stream_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_real_test(ninfer_qwen3_5_tools_real_test NINFER_ARTIFACT_TOOLS
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tools_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_tool_constraints_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_constraints.cpp"
  LIBRARIES ninfer_model_runtime ninfer_grammar ninfer::json)

add_test(NAME ninfer_qwen3_5_tool_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_TOOL_PROBE=$<TARGET_FILE:ninfer_qwen3_5_tool_constraints_test>"
    ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_tool_schema.py")
set_tests_properties(ninfer_qwen3_5_tool_schema_oracle_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_real_test(ninfer_qwen3_5_grammar_real_test NINFER_ARTIFACT_GRAMMAR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_grammar_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_real_test(ninfer_qwen3_5_spec_residue_real_test NINFER_ARTIFACT_RESIDUE
  ARGS --backend mtp --draft-k 7
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_residue_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_real_test(ninfer_qwen3_5_spec_residue_eagle3_real_test NINFER_ARTIFACT_RESIDUE_EAGLE3
  ARGS --backend eagle3 --draft-k 3
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_residue_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_real_test(ninfer_qwen3_5_spec_concurrency_eagle3_real_test NINFER_ARTIFACT_RESIDUE_EAGLE3
  ARGS --backend eagle3 --draft-k 3 --concurrency 2
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_spec_concurrency_real.cpp"
  LIBRARIES ninfer_engine)
