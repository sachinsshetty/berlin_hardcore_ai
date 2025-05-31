# GPU Benchmarking Strategies for Qwen3-0.6B (Berlin Hardcore AI Hackathon)

## Challenge Overview
- **Goal:** Create a benchmark that runs Qwen3-0.6B for 5 minutes, maximizing GPU core usage and producing a single, comparable metric for different NVIDIA GPUs.
- **Key Metrics:** Model throughput (generations/sec or tokens/sec), GPU health (temperature, power, memory usage).

---

## 1. Batching (Increase Parallelism)
- **Description:** Generate multiple outputs in parallel by increasing the batch size.
- **How:**
  - Prepare a list of identical or varied prompts.
  - Tokenize and generate outputs in a single forward pass.
  - Tune batch size to maximize GPU usage without running out of memory.
- **Code Example:**
  ```python
  batch_size = 16  # Adjust based on GPU memory
  prompts = [prompt] * batch_size
  model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
  generated_ids = model.generate(
      **model_inputs,
      max_new_tokens=256,
      do_sample=False,
      use_cache=True,
      pad_token_id=tokenizer.pad_token_id
  )
  ```

---

## 2. Multi-Threading / Multi-Processing
- **Description:** Use Python's threading or multiprocessing to run several generation jobs in parallel, especially if single-process throughput is low.
- **How:**
  - Spawn multiple worker threads/processes, each running a batch generation loop.
  - Aggregate results and statistics.
- **Note:** PyTorch operations are already multi-threaded, but explicit parallelism can help in some cases.

---

## 3. Use High-Performance Inference Libraries
- **vLLM:**
  - Designed for high-throughput, multi-query inference.
  - Can saturate GPU utilization better than vanilla Hugging Face Transformers.
  - [vLLM GitHub](https://github.com/vllm-project/vllm)
- **TensorRT / ONNX Runtime:**
  - Convert model to ONNX and run with TensorRT for optimized inference.
  - May require extra setup and model conversion.

---

## 4. Model Quantization
- **Description:** Use lower-precision (e.g., INT8, FP8) quantized models to fit larger batches in memory and increase throughput.
- **How:**
  - Use quantized model weights if available.
  - Some libraries (e.g., bitsandbytes) support quantized inference.

---

## 5. GPU Monitoring and Health Checks
- **Temperature, Power, Memory:**
  - Use `pynvml` or `nvidia-smi` to monitor and log GPU stats during the benchmark.
  - Report average/max temperature, power usage, and memory consumption.
- **Code Example:**
  ```python
  import pynvml
  pynvml.nvmlInit()
  handle = pynvml.nvmlDeviceGetHandleByIndex(0)
  temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
  power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
  meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
  pynvml.nvmlShutdown()
  ```

---

## 6. Throughput Measurement
- **Generations/sec:** Count how many completions you generate per second.
- **Tokens/sec:** Count total tokens generated per second (preferred for LLMs).
- **How:** Track start/end time, total generations/tokens, and compute averages.

---

## 7. Prompt Engineering for Benchmarking
- Use a fixed, moderately complex prompt to ensure consistent workload.
- Optionally, use a variety of prompts to simulate real-world usage.

---

## 8. Automation and Reporting
- Run the benchmark for a fixed duration (e.g., 5 minutes).
- Output a summary with:
  - Total generations/tokens
  - Average throughput
  - GPU health stats
  - Platform and software versions

---

## 9. Compare Results Across GPUs
- Run the same script on different GPUs (e.g., 5090, H200).
- Use the single throughput metric for comparison.
- The H200 should outperform the 5090 if the benchmark is well-optimized.

---

## 10. Advanced: Custom CUDA Kernels or Model Parallelism
- For expert-level optimization, consider writing custom CUDA kernels or using model parallelism to further saturate GPU resources (rarely needed for this scale).

---

## References
- [Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [PyTorch Inference Tips](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA GPU Monitoring (pynvml)](https://pypi.org/project/pynvml/) 