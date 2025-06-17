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

pip3 install torch torchvision torchaudio \
            --index-url https://pypi.jetson-ai-lab.dev/sbsa/cu128

https://pypi.jetson-ai-lab.dev/

https://github.com/dusty-nv/jetson-containers

https://www.jetson-ai-lab.com/


https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md


Add - daemon.json to /etc/docker/
- sudo systemctl restart docker
-  sudo docker info | grep 'Default Runtime'

- leRobot
 - https://github.com/dusty-nv/jetson-containers/tree/master/packages/robots/lerobot
 - pip install rerun-sdk
 - rerun
 - sudo docker run --runtime nvidia -it --rm --network=host dustynv/lerobot:r36.4.0

 - OpenVLA
  - https://github.com/dusty-nv/jetson-containers/tree/master/packages/vla/openvla
  - sudo docker run --runtime nvidia -it --rm --network=host dustynv/openvla:r36.3.0

 - vllm
   - https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/vllm
   - sudo docker run --runtime nvidia -it --rm --network=host dustynv/vllm:0.6.6.post1-r36.4.0

 - audiocraft
   - https://github.com/dusty-nv/jetson-containers/tree/master/packages/speech/audiocraft
