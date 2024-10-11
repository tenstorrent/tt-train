# Description: Initialize the repository with the necessary configurations
git config --local core.hooksPath .githooks/
chmod +x .githooks/pre-commit
sudo apt install clang-tidy=1:10.0-50~exp1
sudo apt install clang-format=1:10.0-50~exp1
sudo apt  install direnv=2.21.2-1
chmod +x init_tt_metal.sh
source ./init_tt_metal.sh