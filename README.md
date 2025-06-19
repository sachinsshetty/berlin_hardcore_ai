GPUs Brrr - dwani.ai melting GH200 

Results - 

gh200
vllm - 179 tok/sec , 181 tok/sec
llama.cpp - 
- Qwen/Qwen3-0.6B  
    - Tokens per second: 206.24, GPU Utilization (%): 40
    - Tokens per second: 206.89, GPU Utilization (%): 41
- google/gemma-3-1b-it  - 
    - Tokens per second: 42.78 ,GPU Utilization (%): 53
    - Tokens per second: 44.00 ,GPU Utilization (%): 
  

h100

5090


```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/simple_benchmark.py
```


- Project Setup
  - Setup llama.cpp to run Qwen3-0.6B [llama.cpp](docs/llama_cpp_setup.md)
  - Setup tensorRT-LLm [tensorRT-LLM](docs/tensorRT-setup.md)
  - Setup [vllm](vllm.md)


[![Pitch Video](https://img.youtube.com/vi/4DnyKMTQf2w/hqdefault.jpg)](https://youtu.be/4DnyKMTQf2w)

