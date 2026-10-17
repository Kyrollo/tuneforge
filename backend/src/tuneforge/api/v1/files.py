from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/files", tags=["files"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FILE_REGISTRY: dict[str, dict[str, Any]] = {}


class FileMetadata(BaseModel):
    id: str
    filename: str
    size_bytes: int
    content_type: str
    uploaded_at: str
    processed: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


@router.post("/upload", response_model=FileMetadata)
async def upload_file(file: UploadFile = File(...)) -> FileMetadata:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    ext = Path(file.filename).suffix.lower()
    allowed_extensions = {".jsonl", ".json", ".csv", ".txt", ".parquet"}
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    file_id = f"file_{uuid.uuid4().hex[:10]}"
    dest_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    content = await file.read()
    dest_path.write_bytes(content)

    meta = {
        "id": file_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "content_type": file.content_type or "application/octet-stream",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "processed": False,
        "stored_path": str(dest_path),
        "details": {"extension": ext},
    }
    FILE_REGISTRY[file_id] = meta

    return FileMetadata(**meta)


@router.get("", response_model=list[FileMetadata])
async def list_files() -> list[FileMetadata]:
    return [FileMetadata(**meta) for meta in FILE_REGISTRY.values()]


@router.get("/{file_id}", response_model=FileMetadata)
async def get_file(file_id: str) -> FileMetadata:
    if file_id not in FILE_REGISTRY:
        raise HTTPException(status_code=404, detail="File record not found")
    return FileMetadata(**FILE_REGISTRY[file_id])


@router.delete("/{file_id}")
async def delete_file(file_id: str) -> dict[str, str]:
    if file_id not in FILE_REGISTRY:
        raise HTTPException(status_code=404, detail="File record not found")

    stored_path = Path(FILE_REGISTRY[file_id]["stored_path"])
    if stored_path.exists():
        stored_path.unlink()

    del FILE_REGISTRY[file_id]
    return {"status": "deleted", "id": file_id}


@router.post("/{file_id}/process")
async def process_file(file_id: str) -> dict[str, Any]:
    if file_id not in FILE_REGISTRY:
        raise HTTPException(status_code=404, detail="File record not found")

    meta = FILE_REGISTRY[file_id]
    file_path = Path(meta["stored_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored file binary missing")

    ext = Path(meta["filename"]).suffix.lower()
    samples: list[dict[str, Any]] = []

    if ext == ".jsonl":
        for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line_str = line.strip()
            if line_str:
                try:
                    samples.append(json.loads(line_str))
                except Exception:
                    samples.append({"raw": line_str})
    elif ext == ".json":
        try:
            parsed = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(parsed, list):
                samples = parsed
            elif isinstance(parsed, dict):
                samples = parsed.get("data") or parsed.get("examples") or [parsed]
        except Exception:
            samples = [{"raw": "invalid json"}]
    elif ext == ".csv":
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                samples = [dict(row) for row in reader]
        except Exception:
            samples = []
    else:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        samples = [{"text": line} for line in lines if line.strip()]

    meta["processed"] = True
    meta["details"]["sample_count"] = len(samples)

    return {
        "status": "success",
        "file_id": file_id,
        "filename": meta["filename"],
        "sample_count": len(samples),
        "preview": samples[:3],
        "message": f"Successfully parsed {len(samples)} training samples into TuneForge dataset catalog.",
    }
