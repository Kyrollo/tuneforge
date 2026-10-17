from __future__ import annotations

from fastapi import APIRouter
from .health import router as health_router
from .auth import router as auth_router
from .jobs import router as jobs_router
from .files import router as files_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(auth_router)
api_v1_router.include_router(jobs_router)
api_v1_router.include_router(files_router)


@api_v1_router.get("/info", tags=["info"])
async def get_system_info() -> dict[str, str]:
    return {
        "name": "TuneForge",
        "slug": "tuneforge",
        "archetype": "ML lab / experiment tracker",
        "version": "1.0.0",
        "description": "LLM Fine-Tuning Studio: dataset curation, QLoRA/DPO training pipelines, evaluation, GGUF export, and model registry",
    }
