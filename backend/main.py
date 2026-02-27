import sys
import os
import logging

# Ensure the root folder containing 'backend' is in pythonpath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.ml_routes import router as ml_router
from backend.classifier.classifier_routes import router as classifier_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GST Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router)
app.include_router(classifier_router)

@app.get("/")
def read_root():
    return {"message": "GST Fraud Detection API is running"}