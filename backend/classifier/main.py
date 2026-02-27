# backend/main.py — add these lines

from api.classifier_routes import router as classifier_router

app.include_router(classifier_router)