# TuneForge System Architecture & LLM Fine-Tuning Studio Design

TuneForge is an enterprise **Parameter-Efficient LLM Fine-Tuning Studio and Model Registry**. It orchestrates QLoRA (Quantized Low-Rank Adaptation) and DPO (Direct Preference Optimization) training workflows, computes mathematical loss trajectories, registers model checkpoints, and exports quantized GGUF adapters for local deployment.

---

## 1. C4 Container Diagram

```mermaid
flowchart TD
    subgraph Clients["AI / ML Engineering Teams"]
        MLEngineer["Machine Learning Engineer"]
        DatasetPipeline["Data Engineering ETL Pipeline"]
    end

    subgraph TuneForgePlatform["TuneForge Fine-Tuning Platform"]
        UI["TuneForge Web Console (React 18 + TailwindCSS)\nLoss Curves, VRAM Calculator & Inference Comparison Lab"]
        APIGW["FastAPI ML Gateway\nAsync FastAPI (Python 3.12)"]
        
        subgraph CoreEngines["Core Fine-Tuning & Model Management Engines"]
            TrainerEngine["QLoRA Training & Optimization Engine\nParameter Count, Loss Trajectories & Convergence"]
            RegistryEngine["Adapter Registry & GGUF Exporter\nModel Lineage, Checkpoints & Ollama Modelfiles"]
            Store["Rolling Job Store\nPre-seeded Enterprise Model Experiments"]
        end
    end

    subgraph LocalHardware["Workstation GPU Hardware Layer"]
        OllamaLocal["Local Ollama Engine\nDedicated NVIDIA GPU (16GB VRAM) / qwen2.5:0.5b"]
    end

    MLEngineer -->|Monitors Loss Curves & Tests Inference| UI
    DatasetPipeline -->|Uploads Domain Datasets| APIGW
    UI -->|REST Queries /api/v1| APIGW
    APIGW --> TrainerEngine
    APIGW --> RegistryEngine
    TrainerEngine --> Store
    RegistryEngine --> Store
    RegistryEngine -->|Executes Live Comparison| OllamaLocal
```

---

## 2. Fine-Tuning & Model Export Sequence

```mermaid
sequenceDiagram
    autonumber
    actor ML as ML Engineer
    participant UI as TuneForge UI
    participant API as FastAPI Gateway
    participant Trainer as QLoRA Trainer Engine
    participant Registry as Model Registry
    participant GPU as Local Dedicated NVIDIA GPU (Ollama)
    participant Store as Rolling Job Store

    ML->>UI: Submit Dataset & Hyperparameters (r=32, lr=2e-4)
    UI->>API: POST /api/v1/jobs {job_name, base_model, dataset}
    API->>Trainer: Calculate Trainable LoRA Matrices & VRAM Footprint
    Trainer->>Trainer: Compute Exponential Loss Trajectory & Perplexity
    Trainer-->>API: FineTuningJob (Status: COMPLETED, GGUF Exported)
    API->>Store: Persist Job & Register Adapter Checkpoint
    API-->>UI: 200 OK (FineTuningJob)
    UI->>ML: Render Loss Curve Chart & Model Lineage
    ML->>UI: Run Live Dual Inference Comparison
    UI->>API: POST /api/v1/compare {prompt, adapter_id}
    API->>GPU: Execute Prompt with Adapter Conditioning on Dedicated NVIDIA GPU-->>API: Return Generated Response & Latency Telemetry
    API-->>UI: Render Side-by-Side Base vs Fine-Tuned Outputs
```

---

## 3. Core Architectural Modules

### A. QLoRA Training Engine (`trainer_engine.py`)
- Calculates exact low-rank dimension weights for target projection matrices:
  $$N_{\text{trainable}} = 2 \times d_{\text{model}} \times r \times N_{\text{layers}} \times N_{\text{targets}}$$
- Estimates 4-bit NF4 memory footprints, confirming local execution fit on 16GB GPUs without host memory swapping.
- Computes deterministic power-law loss convergence curves:
  $$L(s) = L_{\infty} + (L_0 - L_{\infty}) e^{-\kappa s}$$

### B. Adapter Registry & Exporter (`registry_engine.py`)
- Tracks adapter checkpoints, parent base models, rank $r$, scaling factor $\alpha$, and final perplexity ($PPL = \exp(L_{\text{eval}})$).
- Formats GGUF metadata for immediate deployment to Ollama, llama.cpp, and vLLM runtimes.
- Integrates with workstation local Ollama instance for live side-by-side verification.
