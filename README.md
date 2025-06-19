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

T4 


```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/simple_benchmark.py
```



- Create vLLM container for arm64

    ```bash
    git clone https://github.com/dusty-nv/jetson-containers
    bash jetson-containers/install.sh
    jetson-containers build vllm

    sudo docker tag vllm:r36.4-cu128-24.04-flashinfer vllm:latest

    sudo docker build -t slabstech/dwani-vllm:latest -f Dockerfile .

    sudo docker push slabstech/dwani-vllm:latest
    ```

- Run gemma3-4B-it on vllm
    ```bash
    sudo docker run --runtime nvidia -it --rm -p 7890:8000 slabstech/dwani-vllm

    export HF_TOKEN='some-HF_token-with-gemma-access'
    vllm serve google/gemma-3-4b-it     --served-model-name gemma3     --host 0.0.0.0     --port 7890     --gpu-memory-utilization 0.9     --tensor-parallel-size 1     --max-model-len 16384     --dtype bfloat16 
    ```


- Project Setup
  - Setup llama.cpp to run Qwen3-0.6B [llama.cpp](docs/llama_cpp_setup.md)
  - Setup tensorRT-LLm [tensorRT-LLM](docs/tensorRT-setup.md)


[![Pitch Video](https://img.youtube.com/vi/4DnyKMTQf2w/hqdefault.jpg)](https://youtu.be/4DnyKMTQf2w)

