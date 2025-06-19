# Melting H200 with dwani.ai API: GPU Brrr Squad Slide Deck

## Slide 1: Title Slide
- **Title**: Melting H200 with dwani.ai API: GPU Brrr Squad
- **Subtitle**: Berlin Hardcore AI Hackathon 2025 | May 31 - June 1, 2025
- **Content**:
  - Team: GPU Brrr Squad
  - Mission: Power dwani.ai with concurrent inference & win Benchmark Blitz with Qwen3-0.6B
- **Visual Changes**:
  - **Background**: Dark gradient (black to deep blue) for a "cool, high-tech" GPU Brrr vibe
  - **Title Font**: Bold, futuristic sans-serif (e.g., "Orbitron" or "Montserrat ExtraBold"), neon green (#00FF99), 48pt
  - **Subtitle & Content**: Clean white text, sans-serif (e.g., "Roboto"), 24pt
  - **Imagery**: Animated 3D H200 GPU with a "melting" effect (dripping ice or neon glow), centered
  - **Logo**: Custom "GPU Brrr" logo with icy blue flames, top-left corner
  - **Accent**: Hackathon logo, bottom-right, with a subtle glow effect
- **Notes**: Grab attention with a dynamic, icy-tech design and bold mission statement.

---

## Slide 2: Challenge Overview
- **Title**: Dual Challenge: Dwani.ai & Benchmark Blitz
- **Content**:
  - **Dwani.ai Goal**: Serve 10, 100, 400, 1000, 10,000 LLM inference requests concurrently
  - **Benchmark Blitz**: Run Qwen3-0.6B for 5 min, maximize GPU core usage, compare RTX 5090 vs. H200, check health
  - **Problem**: Current script underutilizes cores, RTX 5090 falsely outperforms H200
- **Visual Changes**:
  - **Background**: Dark blue (#1A237E) with subtle grid pattern for tech feel
  - **Title**: Neon green (#00FF99), bold, 36pt, "Orbitron"
  - **Content**: White bullet points, 20pt, "Roboto", split into two columns for readability
  - **Imagery**: 
    - Left: Arrow graphic (neon green) showing "10 → 10,000" for concurrency
    - Right: Side-by-side RTX 5090 & H200 icons, red "X" over RTX 5090 for flaw
    - Bottom: Heartbeat line (icy blue) for GPU health
  - **Animation**: Arrows pulse to show scaling, red "X" fades in for emphasis
- **Notes**: Frame the stakes with clear, dynamic visuals for scale and issue.

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
- **Visual Changes**:
  - **Background**: Black with subtle icy blue circuit pattern
  - **Title**: Neon green (#00FF99), 36pt, bold "Orbitron"
  - **Content**: White text, 20pt, "Roboto", organized in three boxes (Dwani.ai, Benchmark, Objectives)
  - **Imagery**:
    - Table: VRAM comparison, color-coded (RTX 5090: red, H100: yellow, H200: green), 18pt
    - Red "X" icon over a faded Qwen3-0.6B logo for benchmark flaw
  - **Animation**: Boxes slide in one-by-one, table highlights H200 in green
- **Notes**: Use color and motion to highlight VRAM limits and benchmark gap.

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
- **Visual Changes**:
  - **Background**: Dark blue (#1A237E) with faint NVIDIA logo watermark
  - **Title**: Neon green (#00FF99), 36pt, "Orbitron"
  - **Content**: White text, 20pt, "Roboto", split into two columns (Dwani.ai, Benchmark)
  - **Imagery**:
    - Flowchart: Models (icons) → TensorRT-LLM (gear) → GPUs (H200 green, RTX 5090 red)
    - Green checkmark over H200, yellow clock over RTX 5090 for time-sharing
  - **Animation**: Flowchart lines pulse from models to GPUs, checkmark glows
- **Notes**: Emphasize H200’s fit and TensorRT-LLM’s power with dynamic flow.

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
- **Visual Changes**:
  - **Background**: Black with icy blue "Brrr" wave pattern
  - **Title**: Neon green (#00FF99), 36pt, "Orbitron"
  - **Content**: White text, 18pt, "Roboto", two columns for clarity
  - **Imagery**:
    - Diagram: Requests (arrows) → Batching (gear) → GPU cores (grid of dots)
    - Icons: Tensor core (green chip), NVLink (blue link), Nsight (magnifier)
  - **Animation**: Arrows flow to cores, icons pulse to show tech in action
- **Notes**: Visualize the tech depth and H200’s melting power.

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
- **Visual Changes**:
  - **Background**: Dark gradient (black to blue) with subtle "melt" drip effect
  - **Title**: Neon green (#00FF99), 36pt, "Orbitron"
  - **Content**: White text, 18pt, "Roboto", bullet points in two columns
  - **Imagery**:
    - Bar chart: Throughput (RTX 5090: 5000 red, H200: 12000 green), 3D style
    - Icons: Utilization (gauge), temp (thermometer), power (bolt), all green
  - **Animation**: Bars rise, icons fill (e.g., gauge to 95%) for impact
- **Notes**: Show H200’s dominance with bold, animated visuals.

---

## Slide 7: Why We’ll Win
- **Title**: GPU Brrr Squad: Unstoppable
- **Content**:
  - **Dwani.ai Solution**: Scales to 10,000 requests, melts H200 with efficiency
  - **Benchmark Fix**: Max core usage, H200 beats RTX 5090
  - **Innovation**: TensorRT-LLM + profiling for performance & health
  - **Team**: Experts in GPU optimization, inference, benchmarking
- **Visual Changes**:
  - **Background**: Black with neon green "Brrr" splash effect
  - **Title**: Neon green (#00FF99), 36pt, "Orbitron"
  - **Content**: White text, 20pt, "Roboto", four spotlight boxes
  - **Imagery**:
    - Trophy icon (gold, glowing) for hackathon win
    - Line graph: Utilization (RTX 5090: 75% red, H200: 95% green)
  - **Animation**: Boxes light up, trophy spins, graph lines draw in
- **Notes**: Sell our edge with bold, winning visuals.

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
- **Visual Changes**:
  - **Background**: Dark blue with icy blue "Brrr" burst effect
  - **Title**: Neon green (#00FF99), 36pt, "Orbitron"
  - **Content**: White text, 20pt, "Roboto", clean bullet list
  - **Imagery**:
    - 3D H200 graphic with neon glow, center stage
    - Hackathon banner, bottom, with subtle shine
    - Contact: "GPU Brrr Squad" in bold, icy blue
  - **Animation**: H200 pulses, banner glows, title zooms in
- **Notes**: End with a bold, energetic call to victory.

---

## Presentation Tips
- **Design**: Consistent theme: dark background (black/blue), neon green (#00FF99) & icy blue (#00BFFF) accents for "Brrr" vibe
- **Visuals**: Use 3D GPU icons, animated charts, and glowing effects; keep simple yet bold
- **Delivery**: 1-2 min per slide, emphasize H200’s power for dwani.ai and benchmark win