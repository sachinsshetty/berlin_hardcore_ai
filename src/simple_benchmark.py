import time
import requests
import pynvml

# Initialize NVIDIA SMI for GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Server configuration
SERVER_URL = "http://localhost:7890/v1/chat/completions"
BATCH_SIZE = 512
MAX_TOKENS = 100
TEMPERATURE = 0.7
DURATION_SECONDS = 300

def benchmark():
    start_time = time.time()
    total_tokens = 0
    headers = {"Content-Type": "application/json"}
    
    while time.time() - start_time < DURATION_SECONDS:
        # Prepare batch requests
        for _ in range(BATCH_SIZE):
            payload = {
                "model": "google/gemma-3-1b-it",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            }
            
            response = requests.post(SERVER_URL, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                total_tokens += data["usage"]["completion_tokens"]
        
        time.sleep(0.01)

    # GPU metrics
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
    
    # Performance metrics
    duration = time.time() - start_time
    tokens_per_second = total_tokens / duration
    
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"GPU Utilization (%): {util.gpu}")
    print(f"GPU Temperature (C): {temp}")
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    benchmark()
