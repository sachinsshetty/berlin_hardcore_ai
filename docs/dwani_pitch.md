# Melting H200 with dwani.ai API

## Problem
- Serve 10, 100, 400, 1000, 10000 LLM inference requests concurrently

## Hardware
| GPU Model    | VRAM          |
|--------------|---------------|
| RTX 5090     | 36 GB VRAM    |
| H100         | 80 GB VRAM    |
| H200         | 96 GB VRAM*   |

*Note: Official H200 specs indicate 141 GB VRAM; 96 GB here is assumed for this scenario.

## Open Weight Models and Approximate VRAM Requirements

| Model                                              | VRAM (GB) Approximate |
|----------------------------------------------------|-----------------------|
| **LLM**                                            |                       |
| google/gemma-3-27b-it-qat-q4_0-gguf                | 18                    |
| Qwen/Qwen3-30B-A3B-GGUF (Q8 quantization)          | 32                    |
| **Translate**                                      |                       |
| ai4bharat/indictrans2-indic-en-1B                   | 4                     |
| ai4bharat/indictrans2-en-indic-1B                   | 4.5                   |
| ai4bharat/indictrans2-indic-indic-1B                 | 5                     |
| ai4bharat/IndicTrans3-beta                           | 9                     |
| **TTS**                                            |                       |
| ai4bharat/IndicF5                                   | 1.4                   |
| **ASR**                                            |                       |
| ai4bharat/indic-conformer-600m-multilingual         | 2.5                   |

---

## Notes

- The total VRAM needed to load all models concurrently is approximately:

  18 + 32 + 4 + 4.5 + 5 + 9 + 1.4 + 2.5 = **76.4 GB**

### Comparison: llama.cpp vs vLLM vs TensorRT-LLM for Your Multi-Model Workload on H100

| Aspect               | llama.cpp                                         | vLLM                                               | TensorRT-LLM                                      |
|----------------------|--------------------------------------------------|---------------------------------------------------|--------------------------------------------------|
| **Performance**      | Slowest among the three; lacks full GPU optimization and tensor core usage; single-GPU only | High throughput with continuous batching; good latency under load; optimized for NVIDIA GPUs | Highest throughput and lowest latency; leverages TensorRT, CUDA, cuDNN, and tensor cores fully |
| **Concurrency**      | Moderate concurrency via multi-session; limited batching and scheduling | High concurrency with dynamic continuous batching; handles 32+ requests smoothly | Excellent concurrency; excels especially in batch mode and multi-GPU setups with tensor parallelism |
| **Multi-GPU Support**| No native support; requires external orchestration | Supports multi-GPU scaling and tensor parallelism | Native multi-GPU support with near-linear scaling on NVLINK-connected GPUs like H100 SXM |
| **Quantization Support** | Supports Q3, Q4, Q5, Q6, GGUF formats; flexible for consumer GPUs | Supports GPTQ, AWQ (4-bit), FP8; lacks some variable bit quantization | Supports FP8 and 4-bit quantized models; optimized for NVIDIA GPUs |
| **Hardware Requirements** | Runs on consumer GPUs and CPUs; accessible for experimentation | Requires powerful NVIDIA GPUs; optimized for CUDA environments | Requires NVIDIA GPUs with TensorRT support; best on H100/A100-class GPUs |
| **Ease of Use & Deployment** | Lightweight, portable, easy to compile and run; minimal dependencies | Python API, OpenAI-compatible, easier deployment for serving | Requires model compilation per GPU/OS; more complex setup but enterprise-ready |
| **Use Case Fit**     | Lightweight experimentation, smaller scale inference | High-throughput serving with good latency; flexible for many users | Enterprise-grade, large-scale serving with maximum performance and scalability |
| **Memory Efficiency**| Moderate; quantization reduces VRAM but no advanced memory optimizations | Good memory management with dynamic batching and KV cache | Highly optimized memory usage with layer fusion and dynamic tensor memory management |
| **Community & Ecosystem** | Largest open-source community footprint | Growing adoption in production serving; active development | NVIDIA-backed with strong enterprise support and ecosystem |

---

## Summary for Your Multi-Model Workload (LLM, ASR, TTS on H100)

- **TensorRT-LLM** is the best fit if you prioritize **maximum throughput, lowest latency, and multi-GPU scalability** on H100 hardware. It is ideal for serving large LLMs (e.g., Qwen 30B, Gemma 27B) alongside ASR and TTS models with efficient GPU utilization.
- **vLLM** offers a strong balance of **high concurrency and ease of deployment** with excellent batching and scheduling, suitable if you want a more flexible, Python-friendly solution with good performance on NVIDIA GPUs.
- **llama.cpp** is suitable for **smaller scale or experimentation**, especially if you want to run models on consumer GPUs or CPU environments, but will be significantly slower and less scalable on H100.

---


- llama.cpp

| Setup      | Method                      | Notes                                         |
|------------|-----------------------------|-----------------------------------------------|
| Single GPU | Multiple llama.cpp sessions  | Shared model weights, independent sessions    |
|            | Quantized models (Q4/Q8)     | Reduces VRAM, improves throughput              |
|            | Batching                    | Group requests to maximize GPU utilization     |
| Multi-GPU  | Data parallel inference      | Replicate model per GPU, distribute requests   |
|            | Model parallelism (external) | Requires custom orchestration, not native llama.cpp |
|            | Load balancing              | Distribute requests evenly across GPUs         |


- vllm

| Aspect           | vLLM                                   | llama.cpp                          |
|------------------|---------------------------------------|-----------------------------------|
| Concurrency      | High (continuous batching, 32+ req)    | Moderate (multi-session, limited) |
| Multi-GPU Support| Yes (tensor parallelism)               | No native support                 |
| Quantization     | GPTQ, AWQ (4-bit), FP8                 | Q3, Q4, Q5, Q6, GGUF             |
| Hardware         | High-end GPUs required                 | Consumer GPUs/CPU capable         |
| Throughput       | Very high under load                   | Lower under high concurrency      |
| API & Deployment | OpenAI API compatible, Python API     | Custom API setup needed           |
| Use Case         | Enterprise, large-scale serving        | Lightweight, experimentation     |

- TensorRT-LLM

| Aspect               | Recommendation                           | Benefit                                  |
|----------------------|----------------------------------------|------------------------------------------|
| Model Optimization   | TensorRT-LLM with FP8 / quantized models | Maximize throughput and reduce latency   |
| Hardware Utilization | Use H100 MIG for workload isolation    | Run multiple models concurrently          |
| Concurrency Handling | Batch requests and queue inference     | Improve GPU utilization and throughput    |
| Model Deployment     | Separate MIG slices or time-shared GPU | Flexibility for diverse model types       |
| Monitoring           | Use profiling and telemetry tools      | Dynamic load balancing and scaling        |

- Best method

| Model Type | Suggested Inference Approach on H100                          | Notes                                                                                      |
|------------|---------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| LLM        | Use TensorRT-LLM with FP8-optimized models (e.g., Qwen 30B, Gemma 27B) | Best throughput and latency; supports large models with efficient memory usage             |
|            | Deploy with MIG partitions if running multiple LLMs concurrently | Isolates workloads, maximizes GPU utilization, and allows parallel inference               |
| Translate  | Use TensorRT-optimized translation models or lightweight quantized models | Translation models are smaller (4-9 GB VRAM); can be run alongside LLMs in separate MIG slices or time-shared |
| TTS        | Run TTS models (e.g., IndicF5) on smaller MIG instances or CPU fallback | TTS models have small VRAM footprint (~1.4 GB); can run on MIG slices or CPU if latency tolerant |
| ASR        | Run ASR models (e.g., IndicConformer 600M) via TensorRT or PyTorch with mixed precision | Moderate VRAM (~2.5 GB); can share GPU with other models using MIG or scheduling           |



## References

- Jan.ai benchmarking TensorRT-LLM vs llama.cpp - https://jan.ai/post/benchmarking-nvidia-tensorrt-llm
- BentoML LLM inference backend comparison - https://www.bentoml.com/blog/benchmarking-llm-inference-backends 
- SqueezeBits evaluation of vLLM vs TensorRT-LLM - https://blog.squeezebits.com/vllm-vs-tensorrtllm-1-an-overall-evaluation-30703
- arXiv LLM inference benchmarking on NVIDIA GPUs -  https://arxiv.org/abs/2411.00136v1
- Hyperbolic.xyz LLM serving frameworks overview - https://hyperbolic.xyz/blog/llm-serving-frameworks
- TensorFuse.io throughput comparison - https://tensorfuse.io/blog/llm-throughput-vllm-vs-sglang  
