# TuneForge REST API Reference

TuneForge exposes an asynchronous REST API under the `/api/v1` namespace for orchestrating parameter-efficient fine-tuning jobs, tracking loss curves, and evaluating model adapters.

---

## Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/info` | Service metadata, architecture archetype, and version |
| `GET` | `/api/v1/healthz` | Kubernetes liveness probe |
| `GET` | `/api/v1/readyz` | Kubernetes readiness probe (validates local GPU LLM upstream) |
| `GET` | `/api/v1/jobs` | Returns all fine-tuning jobs and model adapter records |
| `GET` | `/api/v1/jobs/{job_id}` | Detailed experiment report with step-by-step loss history |
| `GET` | `/api/v1/stats` | ML studio KPIs: active jobs, completed models, VRAM footprint |
| `POST` | `/api/v1/jobs` | Initializes and launches a fine-tuning pipeline |
| `POST` | `/api/v1/compare` | Executes live comparison inference on local GPU Ollama |

---

## 1. List Fine-Tuning Experiments
- **Endpoint**: `GET /api/v1/jobs`
- **Response**: `200 OK`
```json
[
  {
    "job_id": "ft-legal-801",
    "job_name": "sre-incident-rca-qlora",
    "base_model": "qwen2.5:0.5b",
    "dataset_name": "enterprise-sre-postmortems-v2",
    "dataset_samples": 1200,
    "method": "QLORA",
    "quantization": "INT4",
    "lora_rank": 32,
    "lora_alpha": 64,
    "learning_rate": 0.0002,
    "epochs": 3,
    "status": "COMPLETED",
    "current_step": 50,
    "total_steps": 50,
    "final_eval_loss": 1.18,
    "perplexity": 3.25,
    "trainable_params_pct": 0.35,
    "gguf_exported": true,
    "export_path": "models/adapters/ft-legal-801-q4_k_m.gguf",
    "loss_history": [ ... ],
    "created_at": "2026-10-11T09:00:00Z"
  }
]
```

---

## 2. Get Studio Statistics
- **Endpoint**: `GET /api/v1/stats`
- **Response**: `200 OK`
```json
{
  "total_jobs": 4,
  "completed_models": 2,
  "active_training_jobs": 1,
  "avg_perplexity_improvement": 42.5,
  "total_vram_allocated_gb": 8.7,
  "registered_adapters": [
    "ft-legal-801",
    "ft-code-802"
  ]
}
```

---

## 3. Compare Inference (Base vs LoRA)
- **Endpoint**: `POST /api/v1/compare`
- **Request Body**:
```json
{
  "prompt": "Analyze root cause of Kubernetes CrashLoopBackOff",
  "model": "qwen2.5:0.5b",
  "adapter_id": "ft-legal-801",
  "temperature": 0.2,
  "max_tokens": 150
}
```
- **Response**: `200 OK`
```json
{
  "adapter_id": "ft-legal-801",
  "base_model": "qwen2.5:0.5b",
  "fine_tuned_output": "Detailed technical RCA analysis...",
  "baseline_output": "Standard baseline response...",
  "tokens_generated": 150,
  "latency_ms": 3661.6,
  "tokens_per_sec": 41.0
}
```
