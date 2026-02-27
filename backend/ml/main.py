# backend/main.py  — add these lines

from api.ml_routes import router as ml_router

app.include_router(ml_router)