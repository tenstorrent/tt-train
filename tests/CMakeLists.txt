include(CTest)
enable_testing()

add_executable(
  warp_drive_tests
  hello_test.cpp
)
target_link_libraries( warp_drive_tests
  GTest::gtest_main
  warp_drive
)

include(GoogleTest)
gtest_discover_tests(warp_drive_tests)