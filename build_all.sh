# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Run setup script to configure env variables, direnv, clang-tidy and clang-format
chmod +x init_repo.sh
source ./init_repo.sh
# Build metal library
cd 3rd_party/tt-metal
./build_metal.sh -b Release
cd ../..
# Build project
cmake -DCMAKE_BUILD_TYPE=Release -B build -GNinja
cmake --build build --config Release --clean-first
