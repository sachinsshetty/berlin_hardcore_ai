# Berlin Hardcore AI Hackathon: GPU Benchmark Blitz Guide

## Challenge Overview
Benchmark Qwen3-0.6B for 5 minutes to produce a single metric for comparing NVIDIA GPUs. The goal is to maximize GPU core usage and ensure the H200 outperforms the 5090 when properly optimized.

---

## 1. Environment Setup

### Clone llama.cpp (if using GGUF models)
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j2
```

### Python Environment
```bash
python -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Model Download

### Download Qwen3-0.6B (Hugging Face Transformers)
```bash
# No manual download needed; the script will download automatically on first run.
```

### (Optional) Download GGUF Model for llama.cpp
```bash
pip install huggingface_hub
mkdir hf_models
huggingface-cli download google/gemma-3-27b-it-qat-q4_0-gguf --local-dir hf_models/
```

---

## 3. Running the Benchmark

### Using the Provided Script
```bash
cd src
python benchmark_win.py
```
- Adjust `BATCH_SIZE` in `benchmark_win.py` if you encounter out-of-memory errors or want to maximize throughput.
- The script will run for 5 minutes and print:
  - Total generations
  - Total tokens generated
  - Throughput (tokens/sec)
  - GPU temperature, power, and memory stats
  - Platform and CUDA/PyTorch version

---

## 4. Tips for Maximizing GPU Utilization
- **Increase Batch Size:** Gradually raise `BATCH_SIZE` until you reach the GPU's memory limit for best throughput.
- **Use FP16:** The script uses `torch.float16` for faster inference and lower memory usage.
- **Monitor GPU Usage:** Use `nvidia-smi` in another terminal to watch GPU utilization, temperature, and power.
- **Try vLLM:** For even higher throughput, consider using [vLLM](https://github.com/vllm-project/vllm) for multi-query inference (requires separate setup).
- **Profile Bottlenecks:** If throughput is low, profile the script to identify and resolve bottlenecks (e.g., data loading, CPU-GPU transfer).

---

## 5. Comparing Results
- Run the same script on both the 5090 and H200 GPUs.
- The H200 should show higher throughput if the benchmark is optimized.
- Record and report the throughput and GPU stats for each run.

---

## 6. Troubleshooting
- **Out of Memory:** Lower `BATCH_SIZE`.
- **Slow Throughput:** Ensure CUDA is enabled, use FP16, and maximize batch size.
- **Model Download Issues:** Ensure you have internet access and the correct Hugging Face credentials.

---

## 7. References & Further Reading
- [Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [PyTorch CUDA Notes](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA GPU Monitoring (pynvml)](https://pypi.org/project/pynvml/)

---

## 8. Optional: Advanced Optimization
- **Quantization:** Try quantized models (INT8/FP8) for larger batch sizes.
- **ONNX/TensorRT:** Convert the model for inference with ONNX Runtime or TensorRT for further speedup.
- **Custom CUDA Kernels:** For expert users, write custom CUDA kernels or use model parallelism for even higher utilization.

---

## 9. Contact & Support
- For issues, open a GitHub issue or contact the hackathon organizers.
- For business/collaboration: sachin (at) dwani (dot) ai 