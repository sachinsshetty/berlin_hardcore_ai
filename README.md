GPUs Brrr - dwani.ai melting GH200 


- Project Setup
  - - Setup llama.cpp to run Qwen3-0.6B

```bash
apt update && apt upgrade
apt install -y cmake libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j2
```
  - Download models
```bash
python -m venv venv
source venv/bin/activate
pip install huggingface_hub
mkdir hf_models 

huggingface-cli download Qwen/Qwen3-0.6B-GGUF --local-dir hf_models/
```

  - Run llama-server
```bash
./build/bin/llama-server --model hf_models/Qwen3-0.6B-Q8_0.gguf --host 127.0.0.1 --port 7860 --n-gpu-layers 100 --threads 4 --ctx-size 4096 --batch-size 256
```

```bash
curl -X POST http://localhost:7860/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "Qwen3-0.6B-Q8_0", "messages": [ {"role": "user", "content": "Hello, how are you?"} ], "max_tokens": 100, "temperature": 0.7 }'
```

- Reference 
  - https://github.com/yachty66/gpu-benchmark/blob/main/src/gpu_benchmark/benchmarks/qwen3_0_6b.py



- For  Qwen/Qwen3-30B-A3B-GGUF
  - huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF --local-dir hf_models/
  - ./build/bin/llama-server --model ./hf_models/Qwen3-30B-A3B-Q8_0.gguf --host 0.0.0.0 --port 7860 --n-gpu-layers 99 --threads 1 --ctx-size 32768 --batch-size 512

  Qwen3-30B-A3B-Q4_K_M.gguf , Qwen3-0.6B-Q8_0.gguf , Qwen3-30B-A3B-Q5_0.gguf ,Qwen3-30B-A3B-Q4_K_M.gguf, Qwen3-30B-A3B-Q8_0.gguf ,Qwen3-30B-A3B-Q5_K_M.gguf, Qwen3-30B-A3B-Q6_K.gguf


- For - Qwen/Qwen3-4B-GGUF
 - mkdir hf_model_4b
 - huggingface-cli download Qwen/Qwen3-4B-GGUF --local-dir hf_model_4b/
 - Qwen3-4B-Q4_K_M.gguf , Qwen3-4B-Q5_0.gguf, Qwen3-4B-Q8_0.gguf, Qwen3-4B-Q6_K.gguf, Qwen3-4B-Q5_K_M.gguf
-  ./build/bin/llama-server --model ./hf_model_4b/Qwen3-4B-Q4_K_M.gguf --host 0.0.0.0 --port 7860 --n-gpu-layers 99 --threads 1 --ctx-size 32768 --batch-size 512

- For - Qwen/Qwen3-14B-GGUF
```bash
mkdir hf_model_14b
huggingface-cli download Qwen/Qwen3-14B-GGUF --local-dir hf_model_14b/
#   Qwen3-14B-Q5_0.gguf , Qwen3-14B-Q6_K.gguf ,Qwen3-14B-Q4_K_M.gguf , Qwen3-14B-Q5_K_M.gguf , Qwen3-14B-Q8_0.gguf- 16 GB
  - ./build/bin/llama-server --model ./hf_model_14b/Qwen3-14B-Q4_K_M.gguf --host 0.0.0.0 --port 7860 --n-gpu-layers 99 --threads 1 --ctx-size 32768 --batch-size 512
- 22 GB

./build/bin/llama-server --model ./hf_model_14b/Qwen3-14B-Q8_0.gguf --host 0.0.0.0 --port 7860 --n-gpu-layers 99 --threads 1 --ctx-size 32768 --batch-size 512
``` 

<!--

./build/bin/llama-server   --model hf_models/Qwen3-0.6B-Q8_0.gguf    --host 0.0.0.0   --port 7860   --n-gpu-layers 100   --threads 1   --ctx-size 8192   --batch-size 512

-->