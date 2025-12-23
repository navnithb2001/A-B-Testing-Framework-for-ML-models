"""
Router for prediction and feedback endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import hashlib
import time
from datetime import datetime
import joblib

from backend.database import get_db, Prediction, Experiment, Model
from backend.config import settings

router = APIRouter()


class PredictRequest(BaseModel):
    """Request model for making predictions"""
    experiment_id: str
    user_id: str
    features: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    """Request model for submitting ground truth"""
    prediction_id: str
    ground_truth: int
    ground_truth_value: Optional[float] = None


def assign_user_to_variant(user_id: str, experiment_id: str, traffic_split: Dict[str, int]) -> str:
    """
    Consistently assign a user to a variant based on hashing
    Same user always gets same variant for same experiment
    """
    # Create deterministic hash
    hash_input = f"{user_id}:{experiment_id}"
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    
    # Convert to 0-99 bucket
    bucket = hash_value % 100
    
    # Assign based on traffic split
    cumulative = 0
    for variant, allocation in traffic_split.items():
        cumulative += allocation
        if bucket < cumulative:
            return variant
    
    # Fallback to first variant
    return list(traffic_split.keys())[0]


def load_model_artifact(file_path: str, model_type: str):
    """Load a model from disk"""
    if model_type in ["sklearn", "xgboost"]:
        return joblib.load(file_path)
    # Add more model types as needed
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


@router.post("/predict")
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Make a prediction using the appropriate model based on A/B test assignment
    
    - **experiment_id**: ID of the running experiment
    - **user_id**: Unique identifier for the user (for consistent bucketing)
    - **features**: Input features as key-value pairs
    """
    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == request.experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status != "running":
        raise HTTPException(status_code=400, detail=f"Experiment is not running (status: {experiment.status})")
    
    # Assign user to variant
    variant = assign_user_to_variant(request.user_id, request.experiment_id, experiment.traffic_split)
    
    # Get the appropriate model
    if variant == "champion":
        model_id = experiment.champion_model_id
    else:
        model_id = experiment.challenger_model_id
    
    model_record = db.query(Model).filter(Model.id == model_id).first()
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Load and run model
    try:
        model = load_model_artifact(model_record.file_path, model_record.model_type)
        
        # Convert features to model input format
        # Assuming features are in correct order or model handles dict
        feature_values = list(request.features.values())
        
        # Measure latency
        start_time = time.time()
        prediction = model.predict([feature_values])[0]
        latency_ms = (time.time() - start_time) * 1000
        
        # Get probability if available (for classification)
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([feature_values])[0]
            probability = float(max(proba))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Generate prediction ID
    prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
    
    # Log prediction to database
    prediction_record = Prediction(
        prediction_id=prediction_id,
        experiment_id=request.experiment_id,
        user_id=request.user_id,
        model_id=model_id,
        variant=variant,
        features=request.features,
        prediction=int(prediction),
        probability=probability,
        latency_ms=latency_ms,
        timestamp=datetime.utcnow(),
        prediction_metadata=request.metadata
    )
    
    db.add(prediction_record)
    db.commit()
    
    return {
        "prediction": int(prediction),
        "probability": probability,
        "prediction_id": prediction_id,
        "model_used": f"{model_record.name}_{model_record.version}",
        "model_version": model_record.version,
        "variant": variant,
        "latency_ms": round(latency_ms, 2),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Submit ground truth feedback for a prediction
    
    - **prediction_id**: ID of the prediction to update
    - **ground_truth**: Actual outcome (0 or 1 for classification)
    - **ground_truth_value**: Actual value for regression (optional)
    """
    # Find prediction
    prediction = db.query(Prediction).filter(Prediction.prediction_id == request.prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Update with ground truth
    prediction.ground_truth = request.ground_truth
    prediction.ground_truth_value = request.ground_truth_value
    prediction.ground_truth_timestamp = datetime.utcnow()
    
    db.commit()
    
    return {
        "status": "success",
        "message": "Feedback recorded",
        "prediction_id": request.prediction_id
    }
