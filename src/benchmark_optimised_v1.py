import time
import requests
import pynvml
import csv
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize NVIDIA SMI for GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Server configuration
SERVER_URL = "http://localhost:9000/v1/chat/completions"
DURATION_SECONDS = 10  # Shorter for faster sweeps

# Hyperparameter grid
BATCH_SIZES = [64, 128]
MAX_TOKENS_LIST = [128, 256]
TEMPERATURES = [0.5, 0.7]

CSV_FILE = "benchmark_results.csv"

def benchmark(batch_size, max_tokens, temperature):
    start_time = time.time()
    total_tokens = 0
    headers = {"Content-Type": "application/json"}
    
    while time.time() - start_time < DURATION_SECONDS:
        for _ in range(batch_size):
            payload = {
                "model": "gemma3",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            try:
                response = requests.post(SERVER_URL, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    total_tokens += data["usage"]["completion_tokens"]
            except Exception as e:
                continue
        time.sleep(0.01)

    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
    duration = time.time() - start_time
    tokens_per_second = total_tokens / duration if duration > 0 else 0
    return tokens_per_second, util.gpu, temp

def main():
    with open(CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["batch_size", "max_tokens", "temperature", "tokens_per_second", "gpu_util", "gpu_temp"])
        for batch_size, max_tokens, temperature in itertools.product(BATCH_SIZES, MAX_TOKENS_LIST, TEMPERATURES):
            print(f"Testing batch_size={batch_size}, max_tokens={max_tokens}, temperature={temperature}")
            tps, gpu_util, gpu_temp = benchmark(batch_size, max_tokens, temperature)
            writer.writerow([batch_size, max_tokens, temperature, tps, gpu_util, gpu_temp])
    pynvml.nvmlShutdown()

    # Visualization
    import pandas as pd
    df = pd.read_csv(CSV_FILE)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = df['batch_size']
    ys = df['max_tokens']
    zs = df['tokens_per_second']
    ax.scatter(xs, ys, zs, c=zs, cmap='viridis')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Max Tokens')
    ax.set_zlabel('Tokens/sec')
    plt.title('GPU Inference Benchmark')
    plt.show()

if __name__ == "__main__":
    main()
