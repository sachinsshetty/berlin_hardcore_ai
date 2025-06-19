# Melting H200 with dwani.ai API: GPU Brrr Squad Slide Deck

## Slide 1: Title Slide
- **Title**: Melting H200 with dwani.ai API: GPU Brrr Squad
- **Subtitle**: Berlin Hardcore AI Hackathon 2025 | May 31 - June 1, 2025
- **Content**:
  - Team: GPU Brrr Squad
  - Mission: Power dwani.ai with concurrent inference & win Benchmark Blitz with Qwen3-0.6B
- **Visuals**:
  - Bold "GPU Brrr" logo or fiery H200 graphic
  - Hackathon logo, date, and venue (Berlin)
- **Notes**: Grab attention with a dynamic design and clear goals.

---

## Slide 2: Challenge Overview
- **Title**: Dual Challenge: Dwani.ai & Benchmark Blitz
- **Content**:
  - **Dwani.ai Goal**: Serve 10, 100, 400, 1000, 10,000 LLM inference requests concurrently
  - **Benchmark Blitz**: Run Qwen3-0.6B for 5 min, maximize GPU core usage, compare RTX 5090 vs. H200, check health
  - **Problem**: Current script underutilizes cores, RTX 5090 falsely outperforms H200
- **Visuals**:
  - Icons for concurrency (arrows), GPUs (RTX 5090, H200), and health (heartbeat)
- **Notes**: Frame the stakes: scalable inference + accurate benchmarking.

---

## Slide 3: Problem Statement
- **Title**: The Stakes
- **Content**:
  - **Dwani.ai**:
    - Multi-model workload: LLM, Translate, TTS, ASR (76.4 GB VRAM total)
    - Hardware: RTX 5090 (36 GB), H100 (80 GB), H200 (96 GB assumed)
  - **Benchmark**:
    - Qwen3-0.6B underutilizes cores in current script
    - H200 should lead RTX 5090 but doesn’t
  - **Objectives**:
    1. Scale inference for dwani.ai
    2. Fix benchmark for performance & GPU health
- **Visuals**:
  - Table: GPU VRAM (RTX 5090: 36 GB, H200: 96 GB)
  - Red "X" on current benchmark flaw
- **Notes**: Highlight the gap and our dual-purpose fix.

---

## Slide 4: Our Solution
- **Title**: TensorRT-LLM: The Powerhouse
- **Content**:
  - **Framework**: TensorRT-LLM for max throughput, low latency, multi-GPU scaling
  - **Dwani.ai**:
    - Load all models (Gemma 27B, Qwen 30B, etc.) with FP8 quantization
    - H200 fits 76.4 GB, RTX 5090 time-shares
  - **Benchmark**:
    - Qwen3-0.6B, 5-min run, dynamic batching
    - Metrics: Throughput (tokens/s), latency, utilization, temp, power
- **Visuals**:
  - Flowchart: Models → TensorRT-LLM → H200/RTX 5090
  - Green check for H200 fit
- **Notes**: Emphasize H200’s edge and fix for core usage.

---

## Slide 5: Technical Approach
- **Title**: How We Melt the H200
- **Content**:
  - **Dwani.ai Serving**:
    - Dynamic batching (1, 8, 32, 64, 128) for 10-10,000 requests
    - Tensor parallelism for LLMs, MIG for TTS/ASR
    - FP8, layer fusion, paged KV caching
  - **Qwen3 Benchmark**:
    - 5-min run, varied inputs (128-1024 tokens)
    - Maximize cores with TensorRT-LLM
    - Profile: Nsight Systems, nvidia-smi
- **Visuals**:
  - Diagram: Requests → Batching → GPU cores
  - Icons: Tensor cores, NVLink
- **Notes**: Show technical depth and H200 optimization.

---

## Slide 6: Expected Outcomes
- **Title**: Results That Win
- **Content**:
  - **Dwani.ai**:
    - H200 serves all models (76.4 GB < 96 GB)
    - RTX 5090 limited, needs time-sharing
    - Scales to 10,000 requests
  - **Benchmark**:
    - H200 >2x RTX 5090 throughput
    - Utilization >90%, temp <85°C, power stable
  - **Metric**: Tokens/second for GPU ranking
- **Visuals**:
  - Bar chart: Throughput (RTX 5090: 5000, H200: 12000 tokens/s)
  - Health icons: Utilization, temp, power
- **Notes**: Prove H200’s dominance for dwani.ai and benchmark.

---

## Slide 7: Why We’ll Win
- **Title**: GPU Brrr Squad: Unstoppable
- **Content**:
  - **Dwani.ai Solution**: Scales to 10,000 requests, melts H200 with efficiency
  - **Benchmark Fix**: Max core usage, H200 beats RTX 5090
  - **Innovation**: TensorRT-LLM + profiling for performance & health
  - **Team**: Experts in GPU optimization, inference, benchmarking
- **Visuals**:
  - Trophy icon for hackathon win
  - Graph: Utilization (RTX 5090: 75%, H200: 95%)
- **Notes**: Sell our edge: scalability, accuracy, skill.

---

## Slide 8: Call to Action
- **Title**: Join the GPU Brrr Revolution
- **Content**:
  - **Resources**:
    - Provided: RTX 5090 (United Compute Cloud)
    - Needed: H200 access (United Compute team)
    - Software: TensorRT-LLM, Nsight, nvidia-smi
  - **Why Us?**: Deliver for dwani.ai, win Benchmark Blitz, melt the H200!
  - **Let’s Win**: Berlin Hardcore AI Hackathon 2025
- **Visuals**:
  - H200 graphic, hackathon banner
  - Contact: GPU Brrr Squad
- **Notes**: End with a bold ask and victory vibe.

---

## Presentation Tips
- **Design**: Use a consistent theme (e.g., dark background, neon green/blue accents for "Brrr" vibe).
- **Visuals**: Keep charts simple, use icons for clarity, and highlight H200’s edge.
- **Delivery**: Practice 1-2 minutes per slide, emphasize scalability for dwani.ai and H200’s benchmark win.