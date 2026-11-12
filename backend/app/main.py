"""FastAPI application entry point for the local solution demo."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import create_router
from backend.app.evaluation_service import EvaluationService


def create_app(service: EvaluationService | None = None) -> FastAPI:
    svc = service or EvaluationService()
    app = FastAPI(title="Answer Sheet Evaluator", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    outputs_dir = getattr(svc, "outputs", None)
    if outputs_dir and Path(outputs_dir).is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

    app.include_router(create_router(svc))
    return app


app = create_app()
