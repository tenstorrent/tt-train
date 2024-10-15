# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Description: Initialize the repository with the necessary configurations
sudo apt install clang-tidy=1:10.0-50~exp1
sudo apt install pre-commit
pre-commit install
sudo apt  install direnv=2.21.2-1
chmod +x init_tt_metal.sh
source ./init_tt_metal.sh
