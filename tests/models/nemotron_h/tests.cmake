# The hybrid block list, with per-layer expert banks: the widths derived from the checkpoint's own
# numbers, and every input the parser is supposed to refuse.
ninfer_add_test(ninfer_nemotron_h_config_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_config.cpp"
  LIBRARIES ninfer_model_loading)

# The composition's own test: it walks the tiny model block by block against the reference dump at
# NINFER_NEMOTRON_REF, so an unset variable skips rather than fails.
ninfer_add_test(ninfer_nemotron_h_forward_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward.cpp"
  LIBRARIES ninfer_model_runtime)

set_tests_properties(ninfer_nemotron_h_forward_test PROPERTIES SKIP_RETURN_CODE 77)

# The loader's own test: it plans and materializes the 45.28 GB artifact at NINFER_NEMOTRON_ARTIFACT
# and checks every block's bound set, so an unset variable skips rather than fails. It links the
# runtime side too, because the model it binds into lives there.
ninfer_add_test(ninfer_nemotron_h_load_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load.cpp"
  LIBRARIES ninfer_model_runtime)

set_tests_properties(ninfer_nemotron_h_load_test PROPERTIES SKIP_RETURN_CODE 77)

# The Program's own state: the classes it derives from a config and a lane count, the zero frontier a
# lane is handed, and the refusals its computing entry points make. It plans state for a synthetic
# config and needs no artifact; the real checkpoint's numbers are checked from the artifact's cold
# directory when NINFER_NEMOTRON_ARTIFACT is set, and that case prints a skip otherwise. It skips
# (77) only when there is no CUDA device, since the state it reports is allocated.
ninfer_add_test(ninfer_nemotron_h_program_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_program.cpp"
  LIBRARIES ninfer_model_runtime)

set_tests_properties(ninfer_nemotron_h_program_test PROPERTIES SKIP_RETURN_CODE 77)

# The attention state and its route: the K/V geometry one lane owns, the append-before-query order, the
# causal-only attention the model has, and the invisibility of slots no token has written. It needs no
# artifact and runs the real route, so it needs a CUDA device and skips (77) without one.
ninfer_add_test(ninfer_nemotron_h_attention_state_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attention_state.cpp"
  LIBRARIES ninfer_model_runtime)

set_tests_properties(ninfer_nemotron_h_attention_state_test PROPERTIES SKIP_RETURN_CODE 77)

# The family's tokenizer: the checkpoint's own pipeline is accepted and every other family's is
# refused, which needs no artifact at all, and -- with NINFER_NEMOTRON_ARTIFACT set -- the family's own
# word boundaries and ids match the reference `tokenizers` pipeline case by case.
ninfer_add_test(ninfer_nemotron_h_tokenizer_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tokenizer.cpp"
  LIBRARIES ninfer_model_runtime ninfer_artifact)

set_tests_properties(ninfer_nemotron_h_tokenizer_test PROPERTIES SKIP_RETURN_CODE 77)

# The corpus comparison the case list cannot cover: the C++ tokenizer against the `tokenizers` pipeline
# over a real corpus, and the artifact's own tokenizer.json against the checkpoint's copy. It needs
# NINFER_NEMOTRON_CORPUS (a UTF-8 text file) as well as the artifact, and skips (77) without either.
add_test(NAME ninfer_nemotron_h_tokenizer_reference_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_TOKEN_PROBE=$<TARGET_FILE:ninfer_nemotron_h_tokenizer_test>"
    ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_tokenizer_reference.py")

set_tests_properties(ninfer_nemotron_h_tokenizer_reference_test PROPERTIES SKIP_RETURN_CODE 77)

# The frontend's load path: the artifact's own text resources build the frontend, which prepares this
# checkpoint's prompt in both thinking modes. It needs the artifact and skips (77) without it.
ninfer_add_test(ninfer_nemotron_h_frontend_load_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend_load.cpp"
  LIBRARIES ninfer_model_runtime ninfer_artifact)

set_tests_properties(ninfer_nemotron_h_frontend_load_test PROPERTIES SKIP_RETURN_CODE 77)

# The frontend's protocol: the artifact's chat template against transformers' own rendering for both
# thinking modes, the sampling and stop defaults the checkpoint declares, every refusal by name, and
# the output decoder's reasoning and tool-call semantics. It needs the artifact too.
ninfer_add_test(ninfer_nemotron_h_frontend_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  LIBRARIES ninfer_model_runtime ninfer_artifact)

set_tests_properties(ninfer_nemotron_h_frontend_test PROPERTIES SKIP_RETURN_CODE 77)

# The Engine's load path for this family: the artifact is dispatched to this family's own instance,
# the Engine reports ready with the metadata it derived from the artifact's own config, and a
# generation is served: the prompt is prefilled into the lane's state, one token is drawn, and each
# later round decodes one token against the state the prefill left. It needs the artifact
# (NINFER_NEMOTRON_ARTIFACT) and skips (77) without it, and links the Engine, the artifact reader and
# the family's config parser, which is what it uses.
ninfer_add_test(ninfer_nemotron_h_engine_load_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_load.cpp"
  LIBRARIES ninfer_engine ninfer_artifact ninfer_model_loading)

set_tests_properties(ninfer_nemotron_h_engine_load_test PROPERTIES SKIP_RETURN_CODE 77)

# The draft's conditioning: the six residual-stream taps a LiLiCorr drafter consumes, captured on the
# real artifact for every position of a real sequence, checked against the same sequence read in one
# pass from a zero state (the incremental capture and the one-pass one must be the same bytes), and
# dumped beside the target's own greedy stream for the acceptance probe. It needs the artifact
# (NINFER_NEMOTRON_ARTIFACT) and skips (77) without it; NINFER_TAP_DUMP names the directory it writes.
ninfer_add_test(ninfer_nemotron_h_draft_taps_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_draft_taps.cpp"
  LIBRARIES ninfer_model_runtime ninfer_artifact)

set_tests_properties(ninfer_nemotron_h_draft_taps_test PROPERTIES SKIP_RETURN_CODE 77)

# The round itself: the drafter proposes, the target verifies a whole block, the longest matching
# prefix is accepted and the bonus token published -- measured as accepted drafts per round on the
# real artifact, with the round's wall time split between the draft's forward and the target's
# verify pass. The bar is 1.5 drafts per round against the reference implementation's 2.122, and a
# number far below it names the round rather than the drafter. It needs the draft-carrying artifact
# (NINFER_NEMOTRON_ARTIFACT) and skips (77) without it.
ninfer_add_test(ninfer_nemotron_h_draft_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_draft_round.cpp"
  LIBRARIES ninfer_model_runtime ninfer_artifact)

set_tests_properties(ninfer_nemotron_h_draft_round_test PROPERTIES SKIP_RETURN_CODE 77)
