
set(ENV{CPM_SOURCE_CACHE} "${PROJECT_SOURCE_DIR}/.cpmcache")

############################################################################################################################
# Boost
############################################################################################################################

include(${PROJECT_SOURCE_DIR}/cmake/fetch_boost.cmake)

fetch_boost_library(smart_ptr)

############################################################################################################################
# yaml-cpp
############################################################################################################################

CPMAddPackage(
  NAME yaml-cpp
  GITHUB_REPOSITORY jbeder/yaml-cpp
  GIT_TAG 0.8.0
  OPTIONS
    "YAML_CPP_BUILD_TESTS OFF"
    "YAML_CPP_BUILD_TOOLS OFF"
    "YAML_BUILD_SHARED_LIBS OFF"
)

if (yaml-cpp_ADDED)
  target_link_libraries(yaml-cpp PUBLIC ${LIBC++} ${LIBC++ABI})
  target_compile_options(yaml-cpp PUBLIC -stdlib=libc++)
endif()

############################################################################################################################
# googletest
############################################################################################################################

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.13.0
  VERSION 1.13.0
  OPTIONS "INSTALL_GTEST OFF"
)

if (googletest_ADDED)
    target_compile_options(gtest PRIVATE -Wno-implicit-int-float-conversion)
    target_link_libraries(gtest PUBLIC ${LIBC++} ${LIBC++ABI})
    target_compile_options(gtest PUBLIC -stdlib=libc++)
    target_link_libraries(gtest_main PUBLIC ${LIBC++} ${LIBC++ABI})
    target_compile_options(gtest_main PUBLIC -stdlib=libc++)
endif()

############################################################################################################################
# boost-ext reflect : https://github.com/boost-ext/reflect
############################################################################################################################

CPMAddPackage(
  NAME reflect
  GITHUB_REPOSITORY boost-ext/reflect
  GIT_TAG v1.1.1
)

# Include FetchContent module
include(FetchContent)


FetchContent_Declare(
    msgpack
    GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
    GIT_TAG cpp-6.1.0  # You can specify a version tag or branch name
  
)
FetchContent_GetProperties(msgpack)
FetchContent_Populate(msgpack)
# Set options before adding the msgpack directory
set(MSGPACK_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(MSGPACK_BUILD_TESTS OFF CACHE INTERNAL "")
set(MSGPACK_BUILD_DOCS OFF CACHE INTERNAL "")
set(MSGPACK_ENABLE_CXX ON CACHE INTERNAL "")
set(MSGPACK_USE_BOOST OFF CACHE INTERNAL "")
set(MSGPACK_BUILD_HEADER_ONLY ON CACHE INTERNAL "")
set(MSGPACK_ENABLE_SHARED OFF CACHE INTERNAL "")
set(MSGPACK_ENABLE_STATIC OFF CACHE INTERNAL "")
set(MSGPACK_CXX20 ON CACHE INTERNAL "")
set(MSGPACK_NO_BOOST ON CACHE BOOL "Disable Boost in msgpack" FORCE)
# Add the msgpack subdirectory
#add_subdirectory(${msgpack_SOURCE_DIR} ${msgpack_BINARY_DIR} EXCLUDE_FROM_ALL)

# Make msgpack available
FetchContent_MakeAvailable(msgpack)