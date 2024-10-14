# Description: This file is used to initialize the environment variables for the TT-Metal project.
# Called from the init_repo.sh file.

if ! grep -q "eval \"\$(direnv hook bash)\"" ~/.bashrc; then
    echo "eval \"\$(direnv hook bash)\"" >> ~/.bashrc
    source ~/.bashrc
fi
touch .envrc
echo export  ARCH_NAME=wormhole_b0 >> .envrc
echo export  TT_METAL_HOME=${PWD}/3rd_party/tt-metal >> .envrc
echo export  PYTHONPATH=${PWD}/3rd_party/tt-metal >> .envrc
echo export  TT_METAL_ENV=dev >> .envrc
direnv allow .