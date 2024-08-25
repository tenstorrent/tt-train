cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION 3.16)
include(cmake/compilers.cmake)

if (DEFINED ENV{CMAKE_C_COMPILER} AND DEFINED ENV{CMAKE_CXX_COMPILER})
    message(STATUS "Setting C and C++ compiler from environment variables")
    set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
    set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
endif()

if (CMAKE_CXX_COMPILER AND CMAKE_C_COMPILER)
    message(STATUS "Using specifed C++ compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Using specifed C compiler: ${CMAKE_C_COMPILER}")
else()
    message(STATUS "No C or C++ compiler specified, defaulting to Clang-17")
    FIND_AND_SET_CLANG17()
endif()


project(ml-framework-cpp)
CHECK_COMPILERS()

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()
if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME is not set")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG=DEBUG")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)


message(STATUS "c++ Standard: ${CMAKE_CXX_STANDARD}")

include(cmake/dependencies.cmake)

message($ENV{TT_METAL_HOME}/tt_metal/third_party/src/firmware/riscv/$ENV{ARCH_NAME})

add_executable(ml-framework-cpp framework/main.cpp)

target_link_libraries(ml-framework-cpp PRIVATE
    dl z pthread atomic # system libraries, hwloc has no cmake support, find_package won't find it
)
add_library(stdlib INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    find_library(LIBC++ c++)
    find_library(LIBC++ABI c++abi)
    if(NOT LIBC++ OR NOT LIBC++ABI)
        message(FATAL_ERROR "libc++ or libc++abi not found. Make sure you have libc++ and libc++abi installed and in your PATH")
    endif()

    target_link_libraries(ml-framework-cpp PRIVATE ${LIBC++} ${LIBC++ABI})
    target_compile_options(ml-framework-cpp PRIVATE -stdlib=libc++)
else()
    target_link_libraries(ml-framework-cpp PRIVATE stdc++)
    target_compile_options(ml-framework-cpp PRIVATE -fsized-deallocation)
endif()

target_include_directories(ml-framework-cpp PRIVATE
    # Metalium
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/wormhole/wormhole_b0_defines
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/wormhole

    # TTNN
    $ENV{TT_METAL_HOME}/ttnn/cpp
    $ENV{TT_METAL_HOME}/ttnn/cpp/ttnn/deprecated
    $ENV{TT_METAL_HOME}/tt_metal/third_party/magic_enum
    ${reflect_SOURCE_DIR} 
    ${flatbuffers_include_dir} 
    ${nng_include_dir}

)
target_link_directories(ml-framework-cpp PRIVATE
    $ENV{TT_METAL_HOME}/build/lib
)

# TODO: In the future TTNN can optionally not depend on Python
find_package(Python REQUIRED Development)

foreach(lib ${BoostPackages})
    target_include_directories(ml-framework-cpp PRIVATE ${Boost${lib}_SOURCE_DIR}/include)
endforeach()

target_link_libraries(ml-framework-cpp PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(ml-framework-cpp PRIVATE
    # Metalium
    tt_metal
    yaml-cpp
    pthread

    # TTNN
    Python::Python
    $ENV{TT_METAL_HOME}/build/lib/_ttnn.so
)

target_compile_definitions(ml-framework-cpp PRIVATE
    FMT_HEADER_ONLY
)

#target_precompile_headers(ml-framework-cpp PRIVATE pch.hpp)