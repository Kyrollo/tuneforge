from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_jobs(client: AsyncClient) -> None:
    response = await client.get("/api/v1/jobs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 4
    job_ids = [j["job_id"] for j in data]
    assert "ft-legal-801" in job_ids
    assert "ft-code-802" in job_ids
    assert "ft-medical-803" in job_ids
    assert "ft-customer-804" in job_ids


@pytest.mark.asyncio
async def test_get_job_detail(client: AsyncClient) -> None:
    response = await client.get("/api/v1/jobs/ft-legal-801")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "ft-legal-801"
    assert data["status"] == "COMPLETED"
    assert data["gguf_exported"] is True
    assert len(data["loss_history"]) > 0


@pytest.mark.asyncio
async def test_get_job_not_found(client: AsyncClient) -> None:
    response = await client.get("/api/v1/jobs/ft-not-found-000")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_stats(client: AsyncClient) -> None:
    response = await client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_jobs"] >= 4
    assert data["completed_models"] >= 2
    assert "registered_adapters" in data
    assert len(data["registered_adapters"]) > 0


@pytest.mark.asyncio
async def test_create_job_empty_dataset(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/jobs",
        json={"job_name": "empty-test", "dataset_text": "   "},
    )
    assert response.status_code == 400
