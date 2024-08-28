# ML-Framework-CPP
CPP ML Framework prototype

# How To Start
run init_repo.sh

# Some commands to use now

cmake  -DCMAKE_BUILD_TYPE=Debug -B debug

cmake --build debug --config Debug --clean-first

clang-tidy framework/main.cpp -p debug
