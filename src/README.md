GPUs Brrr
- apt update && apt upgrade
- apt install -y cmake libcurl4-openssl-dev

- Setup llama.cpp to run Qwen3-0.6B
  - git clone https://github.com/ggml-org/llama.cpp.git
- cd llama.cpp

cmake -B build -DGGML_CUDA=ON

cmake --build build --config Release -j2


python -m venv venv
source venv/bin/activate
pip install huggingface_hub
mkdir hf_models 


huggingface-cli download Qwen/Qwen3-0.6B-GGUF --local-dir hf_models/

./build/bin/llama-server   --model hf_models/Qwen3-0.6B-Q8_0.gguf    --host 0.0.0.0   --port 7860   --n-gpu-layers 100   --threads 1   --ctx-size 8192   --batch-size 512

curl -X POST http://localhost:7860/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "Qwen3-0.6B-Q8_0", "messages": [ {"role": "user", "content": "Hello, how are you?"} ], "max_tokens": 100, "temperature": 0.7 }'


- https://github.com/yachty66/gpu-benchmark/blob/main/src/gpu_benchmark/benchmarks/qwen3_0_6b.py