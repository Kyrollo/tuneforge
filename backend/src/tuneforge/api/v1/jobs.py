from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, status
from src.tuneforge.domain.entities import (
    CreateJobRequest,
    FineTuningJob,
    InferenceCompareRequest,
    MLRegistryKPIs,
)
from src.tuneforge.services.registry_engine import (
    UpstreamOllamaError,
    run_comparison_inference,
)
from src.tuneforge.services.store import store
from src.tuneforge.services.trainer_engine import create_finetuning_pipeline

router = APIRouter(tags=["jobs"])


@router.get("/jobs", response_model=list[FineTuningJob])
async def list_jobs() -> list[FineTuningJob]:
    """Retrieve all parameter-efficient fine-tuning experiments and model adapter runs."""
    return list(store.jobs.values())


@router.get("/jobs/{job_id}", response_model=FineTuningJob)
async def get_job(job_id: str) -> FineTuningJob:
    """Retrieve detailed fine-tuning job metrics, step-by-step loss curve, and export state."""
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return job


@router.get("/stats", response_model=MLRegistryKPIs)
async def get_stats() -> MLRegistryKPIs:
    """Retrieve ML studio KPIs: active training runs, VRAM allocated, and registered adapters."""
    return store.get_kpis()


@router.post("/jobs", response_model=FineTuningJob)
async def create_job(req: CreateJobRequest) -> FineTuningJob:
    """
    Initializes a new fine-tuning pipeline, computing trainable parameter ratios and loss trajectory.
    """
    if not req.dataset_text.strip():
        raise HTTPException(status_code=400, detail="Dataset text cannot be empty")

    job = create_finetuning_pipeline(req)
    store.jobs[job.job_id] = job
    return job


@router.post("/compare")
async def compare_inference(req: InferenceCompareRequest) -> dict[str, Any]:
    """
    Executes live comparison inference on local GPU Ollama instance between base model
    and fine-tuned domain LoRA adapter.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Inference prompt cannot be empty")

    try:
        return await run_comparison_inference(req)
    except UpstreamOllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "type": "https://tuneforge.ai/errors/upstream-unavailable",
                "title": "Local GPU LLM Engine Unavailable",
                "status": 503,
                "detail": str(exc),
            },
        )
