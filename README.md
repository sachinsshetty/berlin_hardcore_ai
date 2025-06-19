GPUs Brrr - dwani.ai melting GH200 


- gh200

| Model                    | Tokens per Second |
|--------------------------|------------------|
| google/gemma-3-1b-it     | 51.24     |
| google/gemma-3-4b-it     | 179   |
| Qwen/Qwen3-0.6B          | 220.82    |
| Qwen/Qwen3-4B            | 164.24    |


llama.cpp - 

| Model                    | Tokens per Second | GPU Utilization (%) |
|--------------------------|-------------------|---------------------|
| google/gemma-3-1b-it     | 43.39     | 53%**               |
| Qwen/Qwen3-0.6B          | 206.57    | 40.5%*              |


- Results for detailed benchmark
  - [gemma3-4b-it](results/gh200/gemma3-4b-benchmark_results.csv)
  - [qwen3-0.6B](results/gh200/qwen3-0-6B-benchmark_results.csv)
  - [qwen4-4B](results/gh200/qwen3-4B-benchmark_results.csv)

- h100

- 5090


- Server Setup
```bash
sudo docker run --runtime nvidia -it --rm -p 9000:9000 dwani/vllm-arm64:latest

vllm serve google/gemma-3-4b-it --served-model-name gemma3 google/gemma-3-1b-it --host 0.0.0.0 --p
ort 9000 --gpu-memory-utilization 0.8

vllm serve google/gemma-3-1b-it --served-model-name gemma3 google/gemma-3-1b-it --host 0.0.0.0 --p
ort 9000 --gpu-memory-utilization 0.8

vllm serve Qwen/Qwen3-0.6B --served-model-name gemma3 google/gemma-3-1b-it --host 0.0.0.0 --p
ort 9000 --gpu-memory-utilization 0.8

vllm serve Qwen/Qwen3-4B --served-model-name gemma3 google/gemma-3-1b-it --host 0.0.0.0 --p
ort 9000 --gpu-memory-utilization 0.8
```

- Run the simple benchmark

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/simple_benchmark.py
```

- Run the optimised benchmark

```bash
python python src/benchmark_optimised_v1.py 
```

- Project Setup
  - Setup llama.cpp to run Qwen3-0.6B [llama.cpp](docs/llama_cpp_setup.md)
  - Setup tensorRT-LLm [tensorRT-LLM](docs/tensorRT-setup.md)
  - Setup [vllm](vllm.md)


[![Pitch Video](https://img.youtube.com/vi/4DnyKMTQf2w/hqdefault.jpg)](https://youtu.be/4DnyKMTQf2w)



