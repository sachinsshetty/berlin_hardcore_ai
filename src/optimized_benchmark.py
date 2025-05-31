"""
Optimized GPU Benchmark for Qwen3-0.6B
This script implements various optimizations to maximize GPU utilization.
"""

import time
import torch
import pynvml
import platform
import os
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress warnings and unnecessary logs
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Qwen/Qwen3-0.6B"
BATCH_SIZE = 64  # Increased from 32 for better GPU utilization
MAX_NEW_TOKENS = 256
BENCHMARK_DURATION = 300  # 5 minutes
PROMPT = "Write a technical explanation of how GPUs process neural networks, in exactly 100 words."
NUM_THREADS = 4  # Number of threads for input preparation and output processing

def get_platform_info():
    """Get detailed platform information."""
    os_platform = platform.system()
    if os_platform == "Linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.strip().split("=")[1].strip('"')
        except Exception:
            pass
        return f"Linux {platform.release()}"
    elif os_platform == "Windows":
        return f"Windows {platform.release()}"
    elif os_platform == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    else:
        return os_platform


def get_nvml_handle():
    """Get NVML handle with better error handling."""
    pynvml.nvmlInit()
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        try:
            idx = int(cuda_visible_devices.split(',')[0])
            return pynvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception:
            pass
    return pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())


def load_model():
    """Load model with optimized settings."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Use FP16 for faster inference
        device_map="auto",
        use_cache=True,
        low_cpu_mem_usage=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def prepare_batch(tokenizer, prompts, device):
    """Prepare a batch of inputs (can be run in a separate thread)."""
    return tokenizer(prompts, return_tensors="pt", padding=True).to(device)


def process_batch(model, inputs):
    """Process a batch through the model."""
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=inputs['input_ids'][0][0]  # Use first token as pad token
        )
    return outputs


def get_gpu_stats(handle):
    """Get comprehensive GPU statistics."""
    stats = {}
    
    # Temperature
    stats['temp'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    
    # Power usage
    try:
        stats['power'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception:
        stats['power'] = None
    
    # Utilization
    try:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats['gpu_util'] = utilization.gpu
        stats['mem_util'] = utilization.memory
    except Exception:
        stats['gpu_util'] = None
        stats['mem_util'] = None
    
    # Memory info
    try:
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        stats['mem_used'] = meminfo.used / (1024**3)
        stats['mem_total'] = meminfo.total / (1024**3)
    except Exception:
        stats['mem_used'] = None
        stats['mem_total'] = None
    
    # Clock speeds
    try:
        stats['sm_clock'] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        stats['mem_clock'] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    except Exception:
        stats['sm_clock'] = None
        stats['mem_clock'] = None
    
    # Check for throttling
    try:
        throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        stats['throttling'] = throttle_reasons > 0
    except Exception:
        stats['throttling'] = None
    
    return stats


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model()
    handle = get_nvml_handle()
    device = model.device

    prompts = [PROMPT] * BATCH_SIZE
    total_tokens = 0
    total_generations = 0
    gpu_stats = []
    throughput_history = []
    
    # Create thread pool for input preparation
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS)
    
    # Prepare initial batches
    batch_futures = [
        executor.submit(prepare_batch, tokenizer, prompts, device)
        for _ in range(NUM_THREADS)
    ]
    
    print(f"Running benchmark for {BENCHMARK_DURATION//60} minutes with batch size {BATCH_SIZE}...")
    start_time = time.time()
    end_time = start_time + BENCHMARK_DURATION
    
    try:
        while time.time() < end_time:
            # Get GPU stats
            stats = get_gpu_stats(handle)
            gpu_stats.append(stats)
            
            # Log current status every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0:
                print(f"Time: {int(elapsed)}s | "
                      f"Generations: {total_generations} | "
                      f"GPU Util: {stats['gpu_util']}% | "
                      f"Temp: {stats['temp']}°C")
            
            # Get a prepared batch (non-blocking)
            if batch_futures:
                future_done, batch_futures = concurrent.futures.wait(
                    batch_futures, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # Process the completed batch
                for future in future_done:
                    try:
                        # Get prepared inputs
                        inputs = future.result()
                        
                        # Submit a new batch preparation task
                        batch_futures.add(executor.submit(
                            prepare_batch, tokenizer, prompts, device
                        ))
                        
                        # Process the current batch
                        batch_start = time.time()
                        outputs = process_batch(model, inputs)
                        batch_end = time.time()
                        
                        # Count tokens generated (excluding prompt tokens)
                        batch_tokens = (outputs.shape[1] - inputs['input_ids'].shape[1]) * outputs.shape[0]
                        total_tokens += batch_tokens
                        total_generations += BATCH_SIZE
                        
                        # Calculate and record throughput for this batch
                        batch_time = batch_end - batch_start
                        batch_throughput = batch_tokens / batch_time if batch_time > 0 else 0
                        throughput_history.append((time.time(), batch_throughput))
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
    
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
    
    finally:
        # Shutdown thread pool
        executor.shutdown(wait=False)
        
        # Calculate final results
        elapsed = time.time() - start_time
        throughput = total_tokens / elapsed if elapsed > 0 else 0
        
        # Calculate averages from GPU stats
        avg_temp = sum(s['temp'] for s in gpu_stats) / len(gpu_stats) if gpu_stats else 0
        max_temp = max(s['temp'] for s in gpu_stats) if gpu_stats else 0
        
        avg_power = None
        if all(s['power'] is not None for s in gpu_stats) and gpu_stats:
            avg_power = sum(s['power'] for s in gpu_stats) / len(gpu_stats)
        
        avg_util = None
        if all(s['gpu_util'] is not None for s in gpu_stats) and gpu_stats:
            avg_util = sum(s['gpu_util'] for s in gpu_stats) / len(gpu_stats)
        
        # Print results
        print("\n===== Benchmark Results =====")
        print(f"Total generations: {total_generations}")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Throughput: {throughput:.2f} tokens/sec")
        print(f"Average GPU utilization: {avg_util:.2f}%" if avg_util is not None else "Average GPU utilization: N/A")
        print(f"Average GPU temp: {avg_temp:.2f}°C (max: {max_temp}°C)")
        if avg_power:
            print(f"Average GPU power: {avg_power:.2f}W")
        
        # Get final memory info
        try:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU memory: {meminfo.used / (1024**3):.2f}GB used / {meminfo.total / (1024**3):.2f}GB total")
        except Exception:
            pass
        
        print(f"Platform: {get_platform_info()}")
        print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'} | PyTorch: {torch.__version__}")
        print(f"Batch size: {BATCH_SIZE}")
        print("============================\n")
        
        # Cleanup
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
