import time
import torch
import pynvml
import platform
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress warnings and unnecessary logs
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Qwen/Qwen3-0.6B"
BATCH_SIZE = 32  # Tune this based on your GPU memory
MAX_NEW_TOKENS = 256
BENCHMARK_DURATION = 300  # 5 minutes
PROMPT = "Write a technical explanation of how GPUs process neural networks, in exactly 100 words."


def get_platform_info():
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
        low_cpu_mem_usage=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def main():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model()
    handle = get_nvml_handle()

    prompts = [PROMPT] * BATCH_SIZE
    total_tokens = 0
    total_generations = 0
    temp_readings = []
    power_readings = []
    start_time = time.time()
    end_time = start_time + BENCHMARK_DURATION

    print(f"Running benchmark for {BENCHMARK_DURATION//60} minutes with batch size {BATCH_SIZE}...")
    while time.time() < end_time:
        # Monitor GPU stats
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        temp_readings.append(temp)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_readings.append(power)
        except Exception:
            pass

        # Prepare batch
        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
            )
        # Count tokens generated (excluding prompt tokens)
        batch_tokens = (outputs.shape[1] - model_inputs['input_ids'].shape[1]) * outputs.shape[0]
        total_tokens += batch_tokens
        total_generations += BATCH_SIZE

    elapsed = time.time() - start_time
    pynvml.nvmlShutdown()

    avg_temp = sum(temp_readings) / len(temp_readings)
    max_temp = max(temp_readings)
    avg_power = sum(power_readings) / len(power_readings) if power_readings else None
    throughput = total_tokens / elapsed

    print("\n===== Benchmark Results =====")
    print(f"Total generations: {total_generations}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(f"Average GPU temp: {avg_temp:.2f}°C (max: {max_temp}°C)")
    if avg_power:
        print(f"Average GPU power: {avg_power:.2f}W")
    try:
        pynvml.nvmlInit()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory total: {meminfo.total / (1024**3):.2f} GB")
        pynvml.nvmlShutdown()
    except Exception:
        pass
    print(f"Platform: {get_platform_info()}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'} | PyTorch: {torch.__version__}")
    print("============================\n")

if __name__ == "__main__":
    main() 