Pitch Document: Melting H200 with dwani.ai API for Berlin Hardcore AI Hackathon 2025
Team Name: GPU Brrr Squad
Challenge Overview
For the Berlin Hardcore AI Hackathon 2025 (May 31 - June 1, 2025), we address the "Melting H200 with dwani.ai API" challenge, aligning with Challenge 4: Benchmark Blitz: GPU Performance Testing with Qwen3. Our goal is to demonstrate a high-performance solution for dwani.ai, serving 10, 100, 400, 1000, and 10,000 LLM inference requests concurrently on RTX 5090 and H200 GPUs, while benchmarking Qwen3-0.6B to maximize core usage and assess GPU health. We’ll deliver a scalable, optimized system to showcase the H200’s superiority and win the hackathon.
Problem Statement

    Dwani.ai Goal: Serve concurrent LLM inference requests (10 to 10,000) for a multi-model workload (LLM, Translate, TTS, ASR) using RTX 5090 (36 GB VRAM), H100 (80 GB VRAM), and H200 (96 GB VRAM assumed).
    Benchmark Blitz: Run Qwen3-0.6B (https://huggingface.co/Qwen/Qwen3-0.6B) for 5 minutes, producing a comparable metric (tokens/second) to rank GPUs, fully utilizing cores, and checking health. The current script (https://github.com/yachty66/gpu-benchmark/blob/main/src/gpu_benchmark/benchmarks/qwen3_0_6b.py) underutilizes cores, wrongly showing RTX 5090 outperforming H200.
    Objectives:
        Enable dwani.ai to handle massive concurrent inference with optimal GPU utilization.
        Benchmark Qwen3-0.6B to measure performance and validate H200’s edge.
        Assess GPU health (utilization, temperature, power).

Our Solution
We propose a TensorRT-LLM-powered solution for dwani.ai, efficiently serving the multi-model workload (76.4 GB VRAM total) and benchmarking Qwen3-0.6B. This maximizes GPU core usage, leverages H200’s compute power, and delivers a robust, scalable system for both goals.
Technical Approach

    Dwani.ai Inference Solution
        Framework: TensorRT-LLM
            Why: Offers top throughput, lowest latency, and native multi-GPU support, ideal for H200’s NVLink and tensor cores.
            Setup: Load all models (Gemma 27B, Qwen 30B, IndicTrans, IndicF5, IndicConformer) with FP8 quantization to fit within 96 GB (H200).
            Concurrency: Handle 10 to 10,000 requests via dynamic batching and queuing.
        Deployment:
            H200: Total VRAM (76.4 GB) fits within 96 GB. Use tensor parallelism for LLMs (Gemma 27B, Qwen 30B) and MIG slices for smaller models (Translate, TTS, ASR).
            RTX 5090: 36 GB VRAM insufficient; time-share models or offload TTS/ASR to CPU.
            Optimization: Layer fusion, paged KV caching, and FP8 reduce latency and boost throughput.
        Scalability: Batch sizes scale dynamically (e.g., 1, 8, 32, 64, 128) to handle 10,000 requests, leveraging H200’s superior compute.
    Benchmark Design for Qwen3-0.6B
        Duration: 5 minutes per GPU (RTX 5090, H200).
        Workload: Simulate 10 to 10,000 concurrent requests, varied inputs (128-1024 tokens), output (128 tokens) to stress cores.
        Metrics:
            Performance: Throughput (tokens/second), time to first token (TTFT), inter-token latency.
            GPU Health: Utilization (%), temperature (°C), power (W), memory (GB).
        Implementation:
            Use TensorRT-LLM for Qwen3-0.6B (FP8, ~0.6-1 GB VRAM).
            Maximize core usage with dynamic batching and optimized kernels.
            Profile with NVIDIA Nsight Systems and nvidia-smi.
        Fix: Current script underutilizes cores. Our approach uses TensorRT-LLM’s optimizations to ensure H200 outperforms RTX 5090.
    Code Plan
        Dwani.ai Serving:
        python

        from tensorrt_llm import LLM, SamplingParams
        import time, nvidia_smi

        # Load models
        models = {
            "gemma": LLM(model_path="google/gemma-3-27b-it-qat-q4_0-gguf", quantization="fp8"),
            "qwen": LLM(model_path="Qwen/Qwen3-30B-A3B-GGUF", quantization="fp8"),
            "indictrans2-ie": LLM(model_path="ai4bharat/indictrans2-indic-en-1B", quantization="fp8"),
            # Add other models (IndicTrans, IndicF5, IndicConformer)
        }
        sampling_params = SamplingParams(max_tokens=128, temperature=0.7)

        # Serve concurrent requests
        def serve_requests(requests, batch_size):
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            for model_name, model in models.items():
                inputs = generate_inputs(batch_size, min_len=128, max_len=1024)
                outputs = model.generate(inputs, sampling_params)
                util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                print(f"{model_name}: {len(outputs)} requests served, Utilization: {util.gpu}%")
            nvidia_smi.nvmlDeviceShutdown()

        # Test concurrency
        for req in [10, 100, 400, 1000, 10000]:
            serve_requests(req, min(req, 128))  # Cap batch size for stability

        Qwen3 Benchmark:
        python

        from tensorrt_llm import LLM, SamplingParams
        import time, nvidia_smi

        # Load Qwen3-0.6B
        model = LLM(model_path="Qwen/Qwen3-0.6B", quantization="fp8")
        sampling_params = SamplingParams(max_tokens=128, temperature=0.7)

        # Benchmark setup
        batch_sizes = [1, 8, 32, 64, 128]
        duration = 300  # 5 minutes
        results = {}
        start = time.time()
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        for batch in batch_sizes:
            tokens = 0
            while time.time() - start < duration:
                inputs = generate_inputs(batch, min_len=128, max_len=1024)
                outputs = model.generate(inputs, sampling_params)
                tokens += count_tokens(outputs)
                util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                temp = nvidia_smi.nvmlDeviceGetTemperature(handle, 0)
                power = nvidia_smi.nvmlDeviceGetPowerUsage(handle) / 1000  # W
            results[batch] = {
                "throughput": tokens / duration,
                "utilization": util.gpu,
                "temperature": temp,
                "power": power
            }

        # Output
        avg_throughput = sum(r["throughput"] for r in results.values()) / len(results)
        print(f"Benchmark Result: {avg_throughput} tokens/second")
        print(f"GPU Health: Utilization {results[128]['utilization']}%, Temp {results[128]['temperature']}°C, Power {results[128]['power']}W")
        nvidia_smi.nvmlDeviceShutdown()

Expected Outcomes

    Dwani.ai: Seamlessly serve 10 to 10,000 requests, with H200 handling all models concurrently (76.4 GB < 96 GB), outperforming RTX 5090 (limited by 36 GB VRAM).
    Benchmark: H200 achieves higher throughput (e.g., >2x RTX 5090) due to superior compute, NVLink, and tensor cores. Utilization >90%, temp <85°C, power stable.
    Metric: Comparable throughput (tokens/second) for GPU ranking, plus health stats.

Why We’ll Win

    Dwani.ai Solution: TensorRT-LLM scales to 10,000 requests, optimizes all models (LLM, Translate, TTS, ASR), and leverages H200’s power for dwani.ai’s needs.
    Benchmark Excellence: Fixes core underutilization, ensuring H200’s superiority in throughput and robust health monitoring.
    Scalability: Handles massive concurrency and adapts to any NVIDIA GPU.
    Innovation: Combines enterprise-grade inference with profiling for a dual-purpose win.

Evaluation Plan

    Test: Run on RTX 5090 (United Compute Cloud) and H200 (coordinate with United Compute team).
    Comparison: H200 shows higher throughput, better utilization than RTX 5090.
    Canvas:
    python

    import matplotlib.pyplot as plt

    # Sample results
    gpus = ["RTX 5090", "H200"]
    throughput = [5000, 12000]  # tokens/second
    utilization = [75, 95]  # %

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(gpus, throughput, color=["blue", "green"])
    ax1.set_title("Throughput (tokens/second)")
    ax1.set_ylabel("Tokens/Second")
    ax2.bar(gpus, utilization, color=["blue", "green"])
    ax2.set_title("GPU Utilization (%)")
    ax2.set_ylabel("Utilization (%)")
    plt.tight_layout()
    plt.show()

Resource Needs

    Provided: RTX 5090 via United Compute Cloud.
    Requested: H200 access via United Compute team.
    Software: TensorRT-LLM, NVIDIA Nsight Systems, nvidia-smi, Python, Hugging Face models.

Team Strengths

    Expertise: GPU optimization, TensorRT-LLM, multi-model serving.
    Impact: Robust dwani.ai solution + winning Qwen3 benchmark.
    Execution: Rapid setup, testing, and visualization for the win.

Why Us?
The GPU Brrr Squad delivers a high-performance, scalable solution for dwani.ai, melting the H200 with concurrent inference, and a top-tier Qwen3 benchmark to prove its might. We’ll win the Berlin Hardcore AI Hackathon 2025 with innovation and excellence!