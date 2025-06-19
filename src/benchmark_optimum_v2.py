import time
import requests
import pynvml
import csv
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import asyncio
import aiohttp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize NVIDIA SMI for GPU monitoring
try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except pynvml.NVMLError as e:
    logging.error(f"Failed to initialize NVML: {e}")
    exit(1)

# Server configuration
SERVER_URL = "http://localhost:9000/v1/chat/completions"
DURATION_SECONDS = 60  # Shorter for faster sweeps

# Hyperparameter grid
BATCH_SIZES = [64, 128]
MAX_TOKENS_LIST = [128, 256]
TEMPERATURES = [0.5, 0.7]

CSV_FILE = "benchmark_results.csv"

async def send_request(session, max_tokens, temperature):
    """Send a single async request to the inference server."""
    payload = {
        "model": "gemma3",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        async with session.post(SERVER_URL, json=payload, timeout=30) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("usage", {}).get("completion_tokens", 0)
            else:
                logging.warning(f"Request failed with status {response.status}")
                return 0
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.warning(f"Request error: {e}")
        return 0

async def benchmark(batch_size, max_tokens, temperature):
    """Benchmark the server with async requests for the given parameters."""
    start_time = time.time()
    total_tokens = 0
    tasks = []
    
    async with aiohttp.ClientSession() as session:
        # Create tasks for batch of requests
        for _ in range(batch_size):
            tasks.append(send_request(session, max_tokens, temperature))
        
        # Run requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, int) or isinstance(result, float):
                total_tokens += result
    
    # Get GPU metrics
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError as e:
        logging.error(f"GPU metric error: {e}")
        util, temp = 0, 0
    
    duration = time.time() - start_time
    tokens_per_second = total_tokens / duration if duration > 0 else 0
    logging.info(f"Batch Size: {batch_size}, Max Tokens: {max_tokens}, Temp: {temperature}, "
                 f"TPS: {tokens_per_second:.2f}, GPU Util: {util.gpu}%, GPU Temp: {temp}C")
    return tokens_per_second, util.gpu, temp

def main():
    try:
        # Write results to CSV
        with open(CSV_FILE, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["batch_size", "max_tokens", "temperature", "tokens_per_second", "gpu_util", "gpu_temp"])
            for batch_size, max_tokens, temperature in itertools.product(BATCH_SIZES, MAX_TOKENS_LIST, TEMPERATURES):
                logging.info(f"Testing batch_size={batch_size}, max_tokens={max_tokens}, temperature={temperature}")
                # Run async benchmark
                tps, gpu_util, gpu_temp = asyncio.run(benchmark(batch_size, max_tokens, temperature))
                writer.writerow([batch_size, max_tokens, temperature, tps, gpu_util, gpu_temp])
    except Exception as e:
        logging.error(f"Error during benchmarking: {e}")
        raise
    finally:
        # Ensure NVML shutdown
        try:
            pynvml.nvmlShutdown()
            logging.info("NVML shutdown successfully")
        except pynvml.NVMLError as e:
            logging.error(f"NVML shutdown failed: {e}")

    # Visualization
    try:
        df = pd.read_csv(CSV_FILE)
        # Check data
        logging.info(f"CSV Data Preview:\n{df.head()}")
        logging.info(f"Data Types:\n{df.dtypes}")
        
        # Convert to numeric, handle errors
        for col in ['batch_size', 'max_tokens', 'temperature', 'tokens_per_second', 'gpu_util', 'gpu_temp']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for missing values
        if df.isnull().any().any():
            logging.warning(f"Missing values detected:\n{df.isnull().sum()}")
        
        # 3D Scatter Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            df['batch_size'],
            df['max_tokens'],
            df['tokens_per_second'],
            c=df['temperature'],
            cmap='viridis',
            s=60
        )
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Max Tokens')
        ax.set_zlabel('Tokens per Second')
        plt.title('GPU Inference Benchmark')
        cbar = plt.colorbar(sc, pad=0.1)
        cbar.set_label('Temperature')
        
        # Save plot
        plt.savefig('gpu_benchmark_3d_plot.png', dpi=300, bbox_inches='tight')
        logging.info("Plot saved as 'gpu_benchmark_3d_plot.png'")
        plt.close()
    except Exception as e:
        logging.error(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()