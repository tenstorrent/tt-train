project (ttml)

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.hpp *.cpp) # I am a bad person

#set(SOURCES)


add_library(ttml STATIC ${SOURCES})

add_library(stdlib INTERFACE)


if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    find_library(LIBC++ c++)
    find_library(LIBC++ABI c++abi)
    if(NOT LIBC++ OR NOT LIBC++ABI)
        message(FATAL_ERROR "libc++ or libc++abi not found. Make sure you have libc++ and libc++abi installed and in your PATH")
    endif()

    target_link_libraries(ttml PUBLIC ${LIBC++} ${LIBC++ABI})
    target_compile_options(ttml PUBLIC -stdlib=libc++)
else()
    target_link_libraries(ttml PUBLIC stdc++)
    target_compile_options(ttml PUBLIC -fsized-deallocation)
endif()

target_include_directories(ttml SYSTEM PUBLIC
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
target_include_directories(ttml PUBLIC ${PROJECT_SOURCE_DIR})
target_link_directories(ttml PUBLIC
    $ENV{TT_METAL_HOME}/build/lib
)
find_package(Python REQUIRED Development)

foreach(lib ${BoostPackages})
    target_include_directories(ttml SYSTEM PUBLIC ${Boost${lib}_SOURCE_DIR}/include)
endforeach()

target_link_libraries(ttml PUBLIC
    dl z pthread atomic
)

target_link_libraries(ttml PUBLIC
    tt_metal
    pthread
    Python::Python
    $ENV{TT_METAL_HOME}/build/lib/_ttnn.so
)

target_link_libraries(ttml PUBLIC 
    yaml-cpp::yaml-cpp
)


target_compile_definitions(ttml PUBLIC
    FMT_HEADER_ONLY
)
