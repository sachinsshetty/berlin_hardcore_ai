VLLM setup

python3.10 -m venv venv

source venv/bin/activate
TMPDIR=/workspace pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

TMPDIR=/workspace pip install vllm huggingface_hub

huggingface-cli download  Qwen/Qwen3-0.6B

vllm serve Qwen/Qwen3-0.6B