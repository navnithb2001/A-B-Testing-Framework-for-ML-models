"""
Router for model management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import uuid
import os
import joblib

from backend.database import get_db, Model
from backend.config import settings

router = APIRouter()


class ModelRegisterRequest(BaseModel):
    """Request model for registering a new ML model"""
    model_name: str
    version: str
    file_path: str
    model_type: str  # sklearn, xgboost, tensorflow, etc.
    alias: Optional[str] = None
    metadata: Optional[dict] = None


class ModelResponse(BaseModel):
    """Response model for model operations"""
    id: str
    name: str
    version: str
    file_path: str
    model_type: str
    alias: Optional[str]
    registered_at: str
    
    class Config:
        from_attributes = True


@router.post("/register", response_model=dict)
async def register_model(request: ModelRegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new ML model in the framework
    
    - **model_name**: Name of the model (e.g., "fraud_detector")
    - **version**: Version identifier (e.g., "v1", "v2")
    - **file_path**: Path to the saved model file
    - **model_type**: Type of model framework
    - **alias**: Optional alias like "champion" or "challenger"
    """
    # Verify model file exists
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {request.file_path}")
    
    # Try to load the model to verify it's valid
    try:
        if request.model_type == "sklearn" or request.model_type == "xgboost":
            joblib.load(request.file_path)
        # Add more model type validations as needed
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")
    
    # Generate model ID
    model_id = f"model_{uuid.uuid4().hex[:8]}"
    
    # Create model entry
    model = Model(
        id=model_id,
        name=request.model_name,
        version=request.version,
        file_path=request.file_path,
        model_type=request.model_type,
        alias=request.alias,
        model_metadata=request.metadata
    )
    
    db.add(model)
    db.commit()
    db.refresh(model)
    
    return {
        "status": "success",
        "model_id": model_id,
        "message": f"Model {request.model_name} v{request.version} registered successfully"
    }


@router.get("/list", response_model=List[ModelResponse])
async def list_models(db: Session = Depends(get_db)):
    """List all registered models"""
    models = db.query(Model).all()
    return models


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get details of a specific model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/{model_id}")
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    """Delete a model from the registry"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if model is used in any active experiments
    # (Add this check later when experiments table is ready)
    
    db.delete(model)
    db.commit()
    
    return {"status": "success", "message": f"Model {model_id} deleted"}
