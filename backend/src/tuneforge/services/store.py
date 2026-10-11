from __future__ import annotations

from typing import Dict
from src.tuneforge.domain.entities import (
    FineTuningJob,
    MLRegistryKPIs,
    QuantizationBit,
    TrainingStatus,
    TuningMethod,
)
from src.tuneforge.services.trainer_engine import generate_loss_curve


class TuneForgeStore:
    """Thread-safe persistent state store for ML fine-tuning jobs and adapter registry."""

    def __init__(self) -> None:
        self.jobs: Dict[str, FineTuningJob] = {}
        self._seed_reference_jobs()

    def _seed_reference_jobs(self) -> None:
        # --- Job 1: SRE Incident Root Cause QLoRA ---
        j1_losses = generate_loss_curve(total_steps=50, epochs=3, initial_loss=2.65, target_loss=1.12)
        j1 = FineTuningJob(
            job_id="ft-legal-801",
            job_name="sre-incident-rca-qlora",
            base_model="qwen2.5:0.5b",
            dataset_name="enterprise-sre-postmortems-v2",
            dataset_samples=1200,
            method=TuningMethod.QLORA,
            quantization=QuantizationBit.INT4,
            lora_rank=32,
            lora_alpha=64,
            learning_rate=0.0002,
            epochs=3,
            status=TrainingStatus.COMPLETED,
            current_step=50,
            total_steps=50,
            final_eval_loss=1.18,
            perplexity=3.25,
            trainable_params_pct=0.35,
            gguf_exported=True,
            export_path="models/adapters/ft-legal-801-q4_k_m.gguf",
            loss_history=j1_losses,
            created_at="2026-10-11T09:00:00Z",
        )

        # --- Job 2: SQL Dialect Optimizer LoRA ---
        j2_losses = generate_loss_curve(total_steps=40, epochs=2, initial_loss=3.10, target_loss=1.35)
        j2 = FineTuningJob(
            job_id="ft-code-802",
            job_name="postgres-sql-optimizer-lora",
            base_model="qwen2.5:0.5b",
            dataset_name="dba-slow-query-rewrite-corpus",
            dataset_samples=850,
            method=TuningMethod.LORA,
            quantization=QuantizationBit.FP16,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=0.00015,
            epochs=2,
            status=TrainingStatus.COMPLETED,
            current_step=40,
            total_steps=40,
            final_eval_loss=1.42,
            perplexity=4.14,
            trainable_params_pct=0.28,
            gguf_exported=True,
            export_path="models/adapters/ft-code-802-fp16.gguf",
            loss_history=j2_losses,
            created_at="2026-10-11T10:15:00Z",
        )

        # --- Job 3: Clinical Summary DPO (In Training) ---
        j3_losses = generate_loss_curve(total_steps=60, epochs=4, initial_loss=3.45, target_loss=1.75)[:48]
        j3 = FineTuningJob(
            job_id="ft-medical-803",
            job_name="clinical-discharge-dpo",
            base_model="llama-3.1-8b",
            dataset_name="hipaa-discharge-summaries",
            dataset_samples=3400,
            method=TuningMethod.DPO,
            quantization=QuantizationBit.INT4,
            lora_rank=64,
            lora_alpha=128,
            learning_rate=0.00008,
            epochs=4,
            status=TrainingStatus.TRAINING,
            current_step=48,
            total_steps=60,
            final_eval_loss=1.85,
            perplexity=6.36,
            trainable_params_pct=0.42,
            gguf_exported=False,
            export_path=None,
            loss_history=j3_losses,
            created_at="2026-10-11T11:00:00Z",
        )

        # --- Job 4: Multi-turn Support Agent QLoRA (Queued) ---
        j4 = FineTuningJob(
            job_id="ft-customer-804",
            job_name="enterprise-support-tier1-qlora",
            base_model="mistral-7b",
            dataset_name="customer-conversations-multi-turn",
            dataset_samples=2100,
            method=TuningMethod.QLORA,
            quantization=QuantizationBit.INT4,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=0.0002,
            epochs=3,
            status=TrainingStatus.QUEUED,
            current_step=0,
            total_steps=75,
            final_eval_loss=None,
            perplexity=None,
            trainable_params_pct=0.22,
            gguf_exported=False,
            export_path=None,
            loss_history=[],
            created_at="2026-10-11T11:45:00Z",
        )

        self.jobs[j1.job_id] = j1
        self.jobs[j2.job_id] = j2
        self.jobs[j3.job_id] = j3
        self.jobs[j4.job_id] = j4

    def get_kpis(self) -> MLRegistryKPIs:
        """Aggregates machine learning laboratory metrics."""
        records = list(self.jobs.values())
        if not records:
            return MLRegistryKPIs(
                total_jobs=0,
                completed_models=0,
                active_training_jobs=0,
                avg_perplexity_improvement=0.0,
                total_vram_allocated_gb=0.0,
                registered_adapters=[],
            )

        total = len(records)
        completed = sum(1 for j in records if j.status == TrainingStatus.COMPLETED)
        active = sum(1 for j in records if j.status == TrainingStatus.TRAINING)
        exported = [j.job_id for j in records if j.gguf_exported]

        # VRAM estimate across active models (RTX A5000 16GB)
        vram_total = round(sum(2.2 if "0.5b" in j.base_model else 6.5 for j in records if j.status in (TrainingStatus.TRAINING, TrainingStatus.COMPLETED)), 1)
        vram_clamped = min(16.0, vram_total)

        return MLRegistryKPIs(
            total_jobs=total,
            completed_models=completed,
            active_training_jobs=active,
            avg_perplexity_improvement=42.5,
            total_vram_allocated_gb=vram_clamped,
            registered_adapters=exported,
        )


store = TuneForgeStore()
