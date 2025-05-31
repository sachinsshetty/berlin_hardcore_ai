VLLM setup

python3.10 -m venv venv

source venv/bin/activate
TMPDIR=/workspace pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128


git clone https://github.com/vllm-project/vllm.git
cd vllm

pip install -e .


// TMPDIR=/workspace pip install vllm huggingface_hub

huggingface-cli download  Qwen/Qwen3-0.6B

vllm serve Qwen/Qwen3-0.6B


- Known issue : SM120 - incompability
    - https://github.com/lllyasviel/Fooocus/issues/3862
