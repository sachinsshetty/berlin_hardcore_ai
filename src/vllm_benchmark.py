"""
vLLM-based GPU Benchmark for Qwen3-0.6B
This script uses vLLM for maximum GPU utilization.
"""

import time
import pynvml
import platform
import os
import torch
from vllm import LLM, SamplingParams

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
BATCH_SIZE = 64  # Adjust based on your GPU memory
MAX_NEW_TOKENS = 256
BENCHMARK_DURATION = 300  # 5 minutes
PROMPT = "Write a technical explanation of how GPUs process neural networks, in exactly 100 words."
GPU_UTILIZATION = 0.9  # Target 90% GPU memory utilization

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
    return pynvml.nvmlDeviceGetHandleByIndex(0)  # Default to first GPU

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
    
    return stats

def main():
    print(f"Loading model: {MODEL_NAME} with vLLM")
    
    # Initialize vLLM with high GPU utilization
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        gpu_memory_utilization=GPU_UTILIZATION,
        tensor_parallel_size=1  # Use 1 for single GPU
    )
    
    # Initialize NVML for GPU monitoring
    handle = get_nvml_handle()
    
    # Define sampling parameters (no randomness for consistent benchmarking)
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic sampling
        max_tokens=MAX_NEW_TOKENS,
        use_beam_search=False
    )
    
    # Prepare prompts
    prompts = [PROMPT] * BATCH_SIZE
    
    # Initialize counters
    total_tokens = 0
    total_generations = 0
    gpu_stats = []
    
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
            
            # Generate text
            outputs = llm.generate(prompts, sampling_params)
            
            # Count tokens
            batch_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            total_tokens += batch_tokens
            total_generations += BATCH_SIZE
    
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
    
    finally:
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
        print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        print(f"Batch size: {BATCH_SIZE}")
        print("============================\n")
        
        # Cleanup
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
