import time
import pynvml
from llama_cpp import Llama

# Initialize NVIDIA SMI for GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Model configuration
MODEL_PATH = "models/Qwen3-0.6B-Q8_0.gguf"  # Update with actual model path
CONTEXT_SIZE = 512
BATCH_SIZE = 512
MAX_TOKENS = 100
TEMPERATURE = 0.7
DURATION_SECONDS = 300

# Initialize llama.cpp model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CONTEXT_SIZE,
    n_gpu_layers=-1,  # Enable full GPU acceleration
    n_batch=BATCH_SIZE,
    n_threads=8,
    verbose=False
)

def benchmark():
    start_time = time.time()
    total_tokens = 0
    prompt = [{"role": "user", "content": "Hello, how are you?"}]
    
    while time.time() - start_time < DURATION_SECONDS:
        # Generate batch completions
        responses = llm.create_chat_completion(
            messages=[prompt] * BATCH_SIZE,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        
        # Handle single and batch responses
        if isinstance(responses, dict):
            responses = [responses]
        
        total_tokens += sum(r["usage"]["completion_tokens"] for r in responses)
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
