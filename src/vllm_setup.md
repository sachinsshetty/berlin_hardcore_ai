VLLM setup

python3.10 -m venv venv

source venv/bin/activate

pip install vllm

pip install huggingface_hub

huggingface-cli download  Qwen/Qwen3-0.6B

vllm serve Qwen/Qwen3-0.6B