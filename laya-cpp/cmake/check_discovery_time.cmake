# Fails when `laya_tests --gtest_list_tests` takes longer than LIMIT seconds. Test registration must
# not load the goldens: ctest runs this listing before every run, with a timeout of its own.
execute_process(COMMAND ${EXE} --gtest_list_tests TIMEOUT ${LIMIT}
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc STREQUAL "0")
  message(FATAL_ERROR "listing tests did not finish in ${LIMIT}s or failed: ${rc}")
endif()
string(LENGTH "${out}" n)
if(n LESS 1000)
  message(FATAL_ERROR "listing produced only ${n} bytes")
endif()
