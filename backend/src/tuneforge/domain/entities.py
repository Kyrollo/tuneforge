from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class TuningMethod(str, Enum):
    QLORA = "QLORA"
    LORA = "LORA"
    FULL_FINETUNE = "FULL_FINETUNE"
    DPO = "DPO"


class QuantizationBit(str, Enum):
    INT4 = "INT4"
    INT8 = "INT8"
    FP16 = "FP16"


class TrainingStatus(str, Enum):
    QUEUED = "QUEUED"
    TRAINING = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class LossStep(BaseModel):
    step: int
    epoch: float
    training_loss: float
    validation_loss: float
    learning_rate: float
    gpu_vram_gb: float


class FineTuningJob(BaseModel):
    job_id: str
    job_name: str
    base_model: str
    dataset_name: str
    dataset_samples: int
    method: TuningMethod
    quantization: QuantizationBit
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 0.0002
    epochs: int = 3
    status: TrainingStatus
    current_step: int
    total_steps: int
    final_eval_loss: float | None = None
    perplexity: float | None = None
    trainable_params_pct: float = 0.28
    gguf_exported: bool = False
    export_path: str | None = None
    loss_history: list[LossStep] = Field(default_factory=list)
    created_at: str


class MLRegistryKPIs(BaseModel):
    total_jobs: int
    completed_models: int
    active_training_jobs: int
    avg_perplexity_improvement: float
    total_vram_allocated_gb: float
    registered_adapters: list[str] = Field(default_factory=list)


class CreateJobRequest(BaseModel):
    job_name: str
    base_model: str = "qwen2.5:0.5b"
    dataset_text: str
    method: TuningMethod = TuningMethod.QLORA
    lora_rank: int = 16
    learning_rate: float = 0.0002
    epochs: int = 3


class InferenceCompareRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:0.5b"
    adapter_id: str = "ft-legal-801"
    temperature: float = 0.2
    max_tokens: int = 180
