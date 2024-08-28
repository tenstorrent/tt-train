include(CTest)
enable_testing()

add_executable(
  ttml_tests
  hello_test.cpp
)
target_link_libraries( ttml_tests
  GTest::gtest_main
  ttml
)

include(GoogleTest)
gtest_discover_tests(ttml_tests)