from __future__ import annotations

import math
import pytest
from src.tuneforge.domain.entities import CreateJobRequest, TrainingStatus, TuningMethod
from src.tuneforge.services.trainer_engine import (
    calculate_trainable_parameters,
    create_finetuning_pipeline,
    generate_loss_curve,
)


def test_calculate_trainable_parameters_qwen() -> None:
    total, trainable, pct, vram = calculate_trainable_parameters("qwen2.5:0.5b", lora_rank=16)
    assert total == 490_000_000
    assert trainable > 0
    assert pct > 0.0 and pct < 1.0  # Parameter-efficient (< 1%)
    assert vram > 1.0 and vram < 5.0  # Runs easily on 16GB Dedicated NVIDIA GPU


def test_calculate_trainable_parameters_llama() -> None:
    total, trainable, pct, vram = calculate_trainable_parameters("llama-3.1-8b", lora_rank=32)
    assert total > 8_000_000_000
    assert trainable > 0
    assert pct < 0.5
    assert vram > 5.0 and vram <= 16.0


def test_generate_loss_curve() -> None:
    steps = generate_loss_curve(total_steps=30, epochs=3, initial_loss=2.5, target_loss=1.2)
    assert len(steps) == 30
    assert steps[0].training_loss > steps[-1].training_loss
    assert steps[0].step == 1
    assert steps[-1].step == 30
    assert steps[-1].validation_loss > 0.0


def test_create_finetuning_pipeline() -> None:
    req = CreateJobRequest(
        job_name="test-incident-qlora",
        base_model="qwen2.5:0.5b",
        dataset_text="Sample 1: Alert triggered on web-01\nSample 2: DB connection exhausted",
        method=TuningMethod.QLORA,
        lora_rank=16,
        learning_rate=0.0002,
        epochs=3,
    )
    job = create_finetuning_pipeline(req)
    assert job.job_name == "test-incident-qlora"
    assert job.status == TrainingStatus.COMPLETED
    assert job.final_eval_loss is not None
    assert job.perplexity is not None
    assert job.perplexity == round(math.exp(job.final_eval_loss), 2)
    assert job.gguf_exported is True
    assert len(job.loss_history) > 0
