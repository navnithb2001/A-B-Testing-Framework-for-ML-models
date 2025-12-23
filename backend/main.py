"""
Main FastAPI application for ML Model A/B Testing Framework
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from backend.database import init_db
from backend.routers import models, experiments, predictions
from backend.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting ML A/B Testing Framework...")
    init_db()
    logger.info("Database initialized successfully")
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="ML Model A/B Testing Framework",
    description="A framework for A/B testing machine learning models in production",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(predictions.router, prefix="/api", tags=["Predictions"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model A/B Testing Framework API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
