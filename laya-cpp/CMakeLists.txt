cmake_minimum_required(VERSION 3.20)
project(laya_cpp VERSION 0.0.0 LANGUAGES CXX)

option(LAYA_BUILD_TESTS "Build the GoogleTest suite" ON)
option(LAYA_ONNX "Build laya_onnx, the ONNX Runtime backend (fetches the pinned runtime unless LAYA_ONNXRUNTIME_ROOT is set)" OFF)
set(LAYA_ONNXRUNTIME_ROOT "" CACHE PATH "An unpacked onnxruntime release (include/ and lib/) to use instead of the download")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Floating point should match the Python reference: no fused multiply-add, no fast math.
# GCC and Clang get -ffp-contract=off. MSVC gets /fp:precise, which only disables fast-math
# reassociation and, before VS2022, does not by itself forbid contraction; the MSVC branch is
# untested and is not exercised by CI until the Windows lane (C11).
if(MSVC)
  set(LAYA_FP_FLAGS /fp:precise)
  set(LAYA_WARN_FLAGS /W4 /WX)
else()
  set(LAYA_FP_FLAGS -ffp-contract=off -fno-fast-math)
  set(LAYA_WARN_FLAGS -Wall -Wextra -Wpedantic -Werror)
endif()

function(laya_target_flags target)
  target_compile_options(${target} PRIVATE ${LAYA_FP_FLAGS} ${LAYA_WARN_FLAGS})
endfunction()

add_library(laya src/version.cpp src/sha256.cpp src/unicodedata.cpp src/pyjson.cpp src/pyvalue.cpp src/lang.cpp
  src/tokenizer.cpp src/schema.cpp src/sequence.cpp src/decode.cpp src/infer.cpp src/collate.cpp src/agent.cpp)
target_include_directories(laya PUBLIC include)
laya_target_flags(laya)

if(LAYA_ONNX)
  include(cmake/onnxruntime.cmake)
  add_library(laya_onnx src/onnx_graph.cpp src/onnx.cpp)
  target_include_directories(laya_onnx PUBLIC include)
  target_link_libraries(laya_onnx PUBLIC laya PRIVATE laya_ort)
  laya_target_flags(laya_onnx)
endif()

if(LAYA_BUILD_TESTS)
  include(FetchContent)
  # Pinned by commit, not tag: tags move. This is the commit of googletest v1.18.0.
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 063de7e9578f82b369302001269680b4b1553359)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(laya_tests tests/version_test.cpp tests/foundation_test.cpp tests/lang_test.cpp tests/hardening_test.cpp
  tests/tokenizer_test.cpp tests/schema_test.cpp tests/sequence_test.cpp tests/decode_test.cpp tests/agent_test.cpp)
  target_link_libraries(laya_tests PRIVATE laya GTest::gtest_main)
  laya_target_flags(laya_tests)

  add_executable(laya_cli tests/tools/laya_cli.cpp)
  target_link_libraries(laya_cli PRIVATE laya)
  laya_target_flags(laya_cli)
  add_executable(laya_c2_cli tests/tools/laya_c2_cli.cpp)
  target_link_libraries(laya_c2_cli PRIVATE laya)
  laya_target_flags(laya_c2_cli)
  add_executable(laya_c3_cli tests/tools/laya_c3_cli.cpp)
  target_link_libraries(laya_c3_cli PRIVATE laya)
  laya_target_flags(laya_c3_cli)
  add_executable(laya_c4_cli tests/tools/laya_c4_cli.cpp)
  target_link_libraries(laya_c4_cli PRIVATE laya)
  laya_target_flags(laya_c4_cli)

  if(LAYA_ONNX)
    add_executable(laya_onnx_tests tests/onnx_test.cpp)
    target_link_libraries(laya_onnx_tests PRIVATE laya_onnx laya_ort GTest::gtest_main)
    laya_target_flags(laya_onnx_tests)
    laya_ort_runtime_path(laya_onnx_tests)
    add_executable(laya_onnx_static_holder tests/tools/onnx_static_holder.cpp)
    target_link_libraries(laya_onnx_static_holder PRIVATE laya_onnx laya_ort)
    laya_target_flags(laya_onnx_static_holder)
    laya_ort_runtime_path(laya_onnx_static_holder)
    add_dependencies(laya_onnx_tests laya_onnx_static_holder)
    add_executable(laya_c5_cli tests/tools/laya_c5_cli.cpp)
    target_link_libraries(laya_c5_cli PRIVATE laya_onnx laya_ort)
    laya_target_flags(laya_c5_cli)
    laya_ort_runtime_path(laya_c5_cli)
    target_compile_definitions(laya_onnx_tests PRIVATE LAYA_ORT_VERSION="${LAYA_ORT_VERSION}"
      LAYA_STATIC_HOLDER="$<TARGET_FILE:laya_onnx_static_holder>")
  endif()

  include(GoogleTest)
  # PRE_TEST: the parametrized rows are counted from the goldens, which exist at test time.
  gtest_discover_tests(laya_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 120)

  if(LAYA_ONNX)
    gtest_discover_tests(laya_onnx_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 120)
    # The refusals of cmake/onnxruntime.cmake (wrong version, missing header or library).
    add_test(NAME ort_root_check COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_ort_root.sh ${LAYA_ORT_ROOT})
  endif()

  # Listing must stay cheap (it runs before every ctest): bound it at 3 s with the goldens loaded.
  add_test(NAME discovery_fast
    COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:laya_tests> -DLIMIT=3
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_discovery_time.cmake)

  # compile_commands.json is the record of what the compiler was really given, so this fails if
  # the floating point flags are dropped from either target.
  # Spelled out here, not derived from LAYA_FP_FLAGS, so dropping a flag above cannot also drop the
  # requirement.
  if(MSVC)
    set(LAYA_FP_FLAGS_ARG "/fp:precise")
  else()
    set(LAYA_FP_FLAGS_ARG "-ffp-contract=off|-fno-fast-math")
  endif()
  add_test(NAME compile_flags
    COMMAND ${CMAKE_COMMAND}
      -DCOMPILE_COMMANDS=${CMAKE_BINARY_DIR}/compile_commands.json
      -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DREQUIRED_FLAGS=${LAYA_FP_FLAGS_ARG}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_compile_flags.cmake)
endif()
