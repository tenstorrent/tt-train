# ML-Framework-CPP
CPP ML Framework prototype

# How To Start
run `source ./init_repo.sh`

# Some commands to use now

cmake  -DCMAKE_BUILD_TYPE=Debug -B debug -GNinja

cmake --build debug --config Debug --clean-first

clang-tidy sources/examples/sample_app/main.cpp -p debug


# TT-Metal
`init_tt_metal.sh` is used to setup metal env variables.
Do not forget to call:
```
git submodule update --init --recursive
```
And manually rebuild tt-metal if needed
