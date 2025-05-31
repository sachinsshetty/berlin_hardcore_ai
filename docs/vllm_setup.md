VLLM setup

python3.10 -m venv venv

source venv/bin/activate


pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128



git clone https://github.com/vllm-project/vllm.git
cd vllm

pip install -e .

// TMPDIR=/workspace pip install vllm huggingface_hub

huggingface-cli download  Qwen/Qwen3-0.6B

vllm serve Qwen/Qwen3-0.6B


- Known issue : SM120 - incompability
    - https://github.com/lllyasviel/Fooocus/issues/3862


-

set TORCH_CUDA_ARCH_LIST=12.0

pip install cmake ninja

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive

Edit setup.py, find TORCH_CUDA_ARCH_LIST, and add:

TORCH_CUDA_ARCH_LIST="8.6+PTX;9.0+PTX;12.0+PTX;12.6+PTX"

MAX_JOBS=8 USE_CUDA=1 python setup.py install

<!-- 
pip install -e . --target /workspace/test

cd workspace
mkdir test
TMPDIR=/workspace pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --target /workspace/test

-->