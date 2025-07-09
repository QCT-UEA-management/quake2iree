

# Add integration test.
add_executable(integration_tests_cpp integration_tests.cpp)
target_link_libraries(integration_tests_cpp PRIVATE lib_test)
target_link_libraries(integration_tests_cpp PUBLIC ${LIBRARY_NAME})
add_test(
  NAME integration_tests_cpp
  COMMAND $<TARGET_FILE:integration_tests_cpp>
)
