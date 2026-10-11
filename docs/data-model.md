# TuneForge Data Model & Machine Learning Schema

The TuneForge data model organizes machine learning experiments into fine-tuning jobs, step-by-step loss trajectories, and adapter registry records.

---

## 1. Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    FINETUNING_JOB ||--|{ LOSS_STEP : "records"
    ML_REGISTRY_KPIS ||--o{ FINETUNING_JOB : "aggregates"

    FINETUNING_JOB {
        string job_id PK "Unique Experiment Identifier"
        string job_name "Experiment run name"
        string base_model "Base foundation model name"
        string dataset_name "Dataset identifier"
        int dataset_samples "Number of instruction samples"
        enum method "QLORA | LORA | FULL_FINETUNE | DPO"
        enum quantization "INT4 | INT8 | FP16"
        int lora_rank "Rank dimension (r)"
        int lora_alpha "Scaling parameter (alpha)"
        float learning_rate "AdamW initial learning rate"
        int epochs "Total training epochs"
        enum status "QUEUED | TRAINING | COMPLETED | FAILED"
        int current_step "Current optimizer step"
        int total_steps "Total scheduled steps"
        float final_eval_loss "Validation loss at convergence"
        float perplexity "Evaluated model perplexity"
        float trainable_params_pct "Percentage of active weights"
        bool gguf_exported "Export checkpoint status"
        string export_path "Filesystem artifact path"
        string created_at "ISO 8601 creation time"
    }

    LOSS_STEP {
        int step PK "Training optimizer step"
        float epoch "Progress in fractional epochs"
        float training_loss "Batch training loss"
        float validation_loss "Evaluation loss"
        float learning_rate "Current learning rate"
        float gpu_vram_gb "Active GPU memory usage"
    }

    ML_REGISTRY_KPIS {
        int total_jobs "Total experiments tracked"
        int completed_models "Converged checkpoints"
        int active_training_jobs "Currently optimizing runs"
        float avg_perplexity_improvement "Mean perplexity gain %"
        float total_vram_allocated_gb "Allocated VRAM footprint"
        json registered_adapters "List of deployed adapter IDs"
    }
```

---

## 2. Quantization & Tuning Method Matrix

| Method | Precision | VRAM Footprint (8B Model) | Trainable Weights % | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| `QLORA` | 4-bit NF4 Base + 16-bit LoRA | **~2.2 GB** | `~0.28%` | On-Premise Workstation GPUs (RTX A5000 16GB) |
| `LORA` | 16-bit FP16 Base + 16-bit LoRA | **~18.5 GB** | `~0.42%` | Single High-End Server GPU (RTX 4090 / A10) |
| `DPO` | 4-bit Reference + Policy Delta | **~4.8 GB** | `~0.35%` | Direct Preference Optimization & Safety Alignment |
| `FULL_FINETUNE` | 16-bit Full Gradient | **~64.0 GB** | `100.0%` | Multi-GPU Distributed Data Center Clusters |
