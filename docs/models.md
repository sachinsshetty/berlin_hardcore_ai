models


- Qwen/Qwen3-0.6B
  - Logs  
    - Tokens per second: 206.24 GPU Utilization (%): 40
    - No Logs
- google/gemma-3-1b-it  - 
  - Logs
    - Tokens per second: 42.78 ,GPU Utilization (%): 53
    - Tokens per second: 44.00 ,GPU Utilization (%): 
  

export VLLM_CONFIGURE_LOGGING=0

 vllm serve Qwen/Qwen3-0.6B   --disable-log-requests --uvicorn-log-level=warning

  vllm serve google/gemma-3-1b-it   --disable-log-requests --uvicorn-log-level=warning