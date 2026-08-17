"""FastAPI application entry point for the local solution demo."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import create_router
from backend.app.evaluation_service import EvaluationService


def create_app(service: EvaluationService | None = None) -> FastAPI:
    app = FastAPI(title="Answer Sheet Evaluator", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(create_router(service or EvaluationService()))
    return app


app = create_app()
