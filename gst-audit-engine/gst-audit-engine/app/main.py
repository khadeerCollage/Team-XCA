"""
GST Audit Report Engine — FastAPI Application Entry Point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="GST Audit Report Engine",
    description=(
        "Deterministic, structured audit report generator for GST compliance validation. "
        "Evaluates the 5-step GST chain (GSTR-1 → GSTR-3B → Tax → IRN → e-Way Bill), "
        "classifies risk, and produces regulator-ready reports."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "GST Audit Report Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
