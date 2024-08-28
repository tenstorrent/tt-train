project (simple_mlp)

set(SOURCES main.cpp)

add_executable(simple_mlp ${SOURCES})
target_link_libraries(simple_mlp PRIVATE ttml)