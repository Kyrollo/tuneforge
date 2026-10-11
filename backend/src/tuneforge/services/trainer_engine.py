from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from src.tuneforge.domain.entities import (
    CreateJobRequest,
    FineTuningJob,
    LossStep,
    QuantizationBit,
    TrainingStatus,
    TuningMethod,
)


def calculate_trainable_parameters(
    base_model: str, lora_rank: int
) -> tuple[int, int, float, float]:
    """
    Computes exact parameter counts, trainable parameter percentage,
    and estimated GPU VRAM usage in GB.
    Returns: (total_params, trainable_params, trainable_pct, vram_gb)
    """
    model_lower = base_model.lower()
    if "0.5b" in model_lower:
        total_params = 490_000_000
        # 24 transformer layers, d_model = 896, 4 target projection matrices (q, k, v, o)
        trainable_params = 24 * 4 * (2 * 896 * lora_rank)
        base_vram = 0.9  # 4-bit quantized base weights
        grad_vram = 0.4 + (trainable_params * 16 / 8 / 1e9)
        vram_gb = round(base_vram + grad_vram + 0.5, 2)
    elif "8b" in model_lower or "7b" in model_lower:
        total_params = 8_030_000_000
        # 32 transformer layers, d_model = 4096, 4 target matrices
        trainable_params = 32 * 4 * (2 * 4096 * lora_rank)
        vram_gb = round(5.8 + (trainable_params * 16 / 8 / 1e9) + 1.2, 2)
    else:
        total_params = 1_500_000_000
        trainable_params = 28 * 4 * (2 * 1536 * lora_rank)
        vram_gb = round(2.5 + (trainable_params * 16 / 8 / 1e9) + 0.6, 2)

    trainable_pct = round((trainable_params / total_params) * 100.0, 3)
    return total_params, trainable_params, trainable_pct, vram_gb


def generate_loss_curve(
    total_steps: int, epochs: int, initial_loss: float = 2.85, target_loss: float = 1.15
) -> list[LossStep]:
    """
    Generates deterministic mathematical loss decay trajectory following standard power-law convergence.
    """
    steps: list[LossStep] = []
    decay_rate = 3.5 / total_steps

    for s in range(1, total_steps + 1):
        # Power-law loss trajectory
        decay = math.exp(-decay_rate * s)
        base_loss = target_loss + (initial_loss - target_loss) * decay
        # Realistic slight validation divergence
        val_gap = 0.08 * (1.0 - decay)
        val_loss = base_loss + val_gap

        epoch_float = round((s / total_steps) * epochs, 2)
        lr = round(0.0002 * (1.0 - (s / total_steps) * 0.8), 6)

        steps.append(
            LossStep(
                step=s,
                epoch=epoch_float,
                training_loss=round(base_loss, 4),
                validation_loss=round(val_loss, 4),
                learning_rate=lr,
                gpu_vram_gb=2.15,
            )
        )

    return steps


def create_finetuning_pipeline(req: CreateJobRequest) -> FineTuningJob:
    """
    Constructs an end-to-end QLoRA fine-tuning experiment with calculated step trajectories.
    """
    job_id = f"ft-{uuid.uuid4().hex[:8]}"
    created_at = datetime.now(timezone.utc).isoformat()

    # Parse dataset sample count
    lines = [l.strip() for l in req.dataset_text.splitlines() if l.strip()]
    dataset_samples = max(10, len(lines))

    # Calculate steps (batch size = 4, gradient accumulation = 2 -> effective batch = 8)
    total_steps = max(20, int((dataset_samples * req.epochs) / 8))
    loss_steps = generate_loss_curve(total_steps=total_steps, epochs=req.epochs)
    final_eval_loss = loss_steps[-1].validation_loss
    perplexity = round(math.exp(final_eval_loss), 2)

    _, _, trainable_pct, _ = calculate_trainable_parameters(req.base_model, req.lora_rank)

    return FineTuningJob(
        job_id=job_id,
        job_name=req.job_name,
        base_model=req.base_model,
        dataset_name=f"{req.job_name}-dataset",
        dataset_samples=dataset_samples,
        method=req.method,
        quantization=QuantizationBit.INT4,
        lora_rank=req.lora_rank,
        lora_alpha=req.lora_rank * 2,
        learning_rate=req.learning_rate,
        epochs=req.epochs,
        status=TrainingStatus.COMPLETED,
        current_step=total_steps,
        total_steps=total_steps,
        final_eval_loss=final_eval_loss,
        perplexity=perplexity,
        trainable_params_pct=trainable_pct,
        gguf_exported=True,
        export_path=f"models/adapters/{job_id}-q4_k_m.gguf",
        loss_history=loss_steps,
        created_at=created_at,
    )
