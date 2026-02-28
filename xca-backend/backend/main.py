"""
main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database.neo4j_connection import get_driver, close_driver
from routers.all_routers import (
    taxpayers_router,
    invoices_router,
    reconcile_router,
    risk_router,
    audit_router,
    graph_router,
)
from routers.pipeline_router import pipeline_router
from routers.ml_router import ml_router
from routers.audit_engine_router import audit_engine_router


# ── Startup / Shutdown lifecycle ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ON STARTUP: initialize DB connection
    print("Connecting to Neo4j...")
    get_driver()
    print("Neo4j connected")
    yield
    # ON SHUTDOWN: close DB connection
    print("Closing Neo4j connection...")
    close_driver()


# ── Create App ────────────────────────────────────────────────
app = FastAPI(
    title       = "GST Reconciliation API",
    description = """
## Intelligent GST Reconciliation Using Knowledge Graphs

Built for **HackWithAI #76** — FinTech / GovTech / Graph AI

### What this API does:
- **Ingests** GST data (GSTR-1, GSTR-2B, e-Invoice, e-Way Bill) into Neo4j
- **Traverses** the graph to detect ITC mismatches (multi-hop)
- **Scores** vendor compliance risk (CIBIL-style)
- **Generates** plain-English audit trails via LLM
- **Exposes** graph data for React dashboard visualization

### Quick Start:
1. `POST /reconcile/run` → Run full reconciliation
2. `GET /reconcile/summary` → Dashboard stats
3. `GET /risk/leaderboard` → Vendor risk scores
4. `GET /graph/nodes-edges` → Graph visualization data
5. `GET /audit/invoice/{id}` → Audit trail for an invoice
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── CORS — allow React frontend to call this API ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Register all routers ──────────────────────────────────────
app.include_router(taxpayers_router)
app.include_router(invoices_router)
app.include_router(reconcile_router)
app.include_router(risk_router)
app.include_router(audit_router)
app.include_router(graph_router)
app.include_router(pipeline_router)
app.include_router(ml_router)
app.include_router(audit_engine_router)


# ── Health check + root ───────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "GST Reconciliation API is running",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    try:
        from database.neo4j_connection import run_query
        run_query("RETURN 1 AS ok")
        db_status = "Connected"
    except Exception as e:
        db_status = f"❌ {str(e)}"

    return {
        "api":      "Running",
        "neo4j":    db_status,
        "version":  "1.0.0",
    }


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
