
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
