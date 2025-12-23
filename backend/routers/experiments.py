"""
Router for experiment management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
from datetime import datetime, timedelta

from backend.database import get_db, Experiment, Model, Prediction
from backend.config import settings

router = APIRouter()


class ExperimentCreateRequest(BaseModel):
    """Request model for creating an A/B test experiment"""
    experiment_name: str
    champion_model_id: str
    challenger_model_id: str
    traffic_split: Dict[str, int] = {"champion": 80, "challenger": 20}
    primary_metric: str = "accuracy"
    secondary_metrics: Optional[List[str]] = None
    duration_days: int = 7
    min_sample_size: Optional[int] = None


class ExperimentResponse(BaseModel):
    """Response model for experiment operations"""
    id: str
    name: str
    champion_model_id: str
    challenger_model_id: str
    traffic_split: Dict[str, int]
    primary_metric: str
    status: str
    start_date: Optional[str]
    end_date: Optional[str]
    created_at: str
    
    class Config:
        from_attributes = True


@router.post("/create", response_model=dict)
async def create_experiment(request: ExperimentCreateRequest, db: Session = Depends(get_db)):
    """
    Create a new A/B test experiment
    
    - **experiment_name**: Descriptive name for the experiment
    - **champion_model_id**: ID of the current production model
    - **challenger_model_id**: ID of the new model to test
    - **traffic_split**: Percentage split (must sum to 100)
    - **primary_metric**: Main metric to optimize (accuracy, precision, recall, etc.)
    - **duration_days**: How long to run the experiment
    """
    # Validate models exist
    champion = db.query(Model).filter(Model.id == request.champion_model_id).first()
    if not champion:
        raise HTTPException(status_code=404, detail="Champion model not found")
    
    challenger = db.query(Model).filter(Model.id == request.challenger_model_id).first()
    if not challenger:
        raise HTTPException(status_code=404, detail="Challenger model not found")
    
    # Validate traffic split
    if sum(request.traffic_split.values()) != 100:
        raise HTTPException(status_code=400, detail="Traffic split must sum to 100")
    
    # Generate experiment ID
    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
    
    # Create experiment
    experiment = Experiment(
        id=experiment_id,
        name=request.experiment_name,
        champion_model_id=request.champion_model_id,
        challenger_model_id=request.challenger_model_id,
        traffic_split=request.traffic_split,
        primary_metric=request.primary_metric,
        secondary_metrics=request.secondary_metrics or [],
        status="running",
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + timedelta(days=request.duration_days),
        min_sample_size=request.min_sample_size or settings.DEFAULT_MIN_SAMPLE_SIZE
    )
    
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    
    return {
        "status": "success",
        "experiment_id": experiment_id,
        "message": f"Experiment '{request.experiment_name}' created and now routing traffic",
        "expected_completion": experiment.end_date.isoformat()
    }


@router.get("/list", response_model=List[ExperimentResponse])
async def list_experiments(status: Optional[str] = None, db: Session = Depends(get_db)):
    """List all experiments, optionally filtered by status"""
    query = db.query(Experiment)
    if status:
        query = query.filter(Experiment.status == status)
    experiments = query.all()
    return experiments


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Get details of a specific experiment"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/{experiment_id}/stop")
async def stop_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Stop a running experiment"""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status != "running":
        raise HTTPException(status_code=400, detail=f"Experiment is not running (status: {experiment.status})")
    
    experiment.status = "stopped"
    experiment.end_date = datetime.utcnow()
    db.commit()
    
    return {
        "status": "success",
        "message": f"Experiment {experiment_id} stopped"
    }


@router.get("/{experiment_id}/results")
async def get_experiment_results(experiment_id: str, db: Session = Depends(get_db)):
    """
    Get comprehensive statistical analysis results for an experiment
    
    Analyzes both model performance metrics (accuracy, precision, recall, F1, latency)
    and statistical significance using Welch's t-test and proportion z-tests
    """
    from ..statistical.metrics import MetricsCalculator
    from ..statistical.tests import (
        welch_t_test, proportions_z_test, confidence_interval,
        confidence_interval_difference, effect_size_cohens_d
    )
    
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Get all predictions for this experiment, grouped by variant
    champion_predictions = db.query(Prediction).filter(
        Prediction.experiment_id == experiment_id,
        Prediction.variant == "champion",
        Prediction.ground_truth.isnot(None)  # Only include predictions with feedback
    ).all()
    
    challenger_predictions = db.query(Prediction).filter(
        Prediction.experiment_id == experiment_id,
        Prediction.variant == "challenger",
        Prediction.ground_truth.isnot(None)
    ).all()
    
    # Check if we have enough data
    if len(champion_predictions) < 5 or len(challenger_predictions) < 5:
        return {
            "experiment_id": experiment_id,
            "status": experiment.status,
            "error": "Insufficient data for statistical analysis",
            "champion_sample_size": len(champion_predictions),
            "challenger_sample_size": len(challenger_predictions),
            "minimum_required": 5,
            "message": "Please collect more predictions with feedback for both variants"
        }
    
    # Extract data for champion
    champion_y_true = [int(p.ground_truth) for p in champion_predictions]  # type: ignore
    champion_y_pred = [int(p.prediction) for p in champion_predictions]  # type: ignore
    champion_latencies = [float(p.latency_ms) for p in champion_predictions]  # type: ignore
    
    # Extract data for challenger
    challenger_y_true = [int(p.ground_truth) for p in challenger_predictions]  # type: ignore
    challenger_y_pred = [int(p.prediction) for p in challenger_predictions]  # type: ignore
    challenger_latencies = [float(p.latency_ms) for p in challenger_predictions]  # type: ignore
    
    # Calculate performance metrics
    calc = MetricsCalculator()
    champion_metrics = calc.classification_metrics(champion_y_true, champion_y_pred)
    challenger_metrics = calc.classification_metrics(challenger_y_true, challenger_y_pred)
    champion_latency = calc.latency_metrics(champion_latencies)
    challenger_latency = calc.latency_metrics(challenger_latencies)
    
    # Statistical tests for accuracy (proportions test)
    accuracy_test = proportions_z_test(
        successes_a=int(champion_metrics["accuracy"] * champion_metrics["sample_size"]),
        n_a=champion_metrics["sample_size"],
        successes_b=int(challenger_metrics["accuracy"] * challenger_metrics["sample_size"]),
        n_b=challenger_metrics["sample_size"]
    )
    
    # Statistical test for latency (Welch's t-test)
    latency_test = welch_t_test(champion_latencies, challenger_latencies)
    
    # Effect sizes
    accuracy_effect_size = effect_size_cohens_d(
        champion_y_true, challenger_y_true
    )
    
    # Confidence intervals
    champion_accuracy_ci = confidence_interval(
        [1 if champion_y_true[i] == champion_y_pred[i] else 0 
         for i in range(len(champion_y_true))]
    )
    challenger_accuracy_ci = confidence_interval(
        [1 if challenger_y_true[i] == challenger_y_pred[i] else 0 
         for i in range(len(challenger_y_true))]
    )
    
    accuracy_diff_ci = confidence_interval_difference(
        [1 if champion_y_true[i] == champion_y_pred[i] else 0 
         for i in range(len(champion_y_true))],
        [1 if challenger_y_true[i] == challenger_y_pred[i] else 0 
         for i in range(len(challenger_y_true))]
    )
    
    # Determine winner and recommendation
    accuracy_diff = challenger_metrics["accuracy"] - champion_metrics["accuracy"]
    latency_diff_pct = ((challenger_latency["mean_ms"] - champion_latency["mean_ms"]) 
                        / champion_latency["mean_ms"] * 100)
    
    # Decision logic
    recommendation = "keep_champion"
    reasoning = []
    
    if accuracy_test["p_value"] < 0.05:
        if accuracy_diff > 0:
            reasoning.append(f"Challenger has significantly better accuracy (+{accuracy_diff*100:.2f}%)")
            recommendation = "promote_challenger"
        else:
            reasoning.append(f"Champion has significantly better accuracy ({abs(accuracy_diff)*100:.2f}% higher)")
    else:
        reasoning.append("No significant difference in accuracy detected")
    
    if latency_test["p_value"] < 0.05:
        if latency_diff_pct > 0:
            reasoning.append(f"Challenger is significantly slower (+{latency_diff_pct:.2f}%)")
            if recommendation == "promote_challenger":
                recommendation = "needs_review"
        else:
            reasoning.append(f"Challenger is significantly faster ({abs(latency_diff_pct):.2f}% reduction)")
            if recommendation == "keep_champion":
                recommendation = "promote_challenger"
    else:
        reasoning.append("No significant difference in latency detected")
    
    return {
        "experiment_id": experiment_id,
        "status": experiment.status,
        "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
        "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
        
        # Sample sizes
        "sample_sizes": {
            "champion": champion_metrics["sample_size"],
            "challenger": challenger_metrics["sample_size"],
            "total": champion_metrics["sample_size"] + challenger_metrics["sample_size"]
        },
        
        # Champion metrics
        "champion": {
            "model_id": experiment.champion_model_id,
            "performance": champion_metrics,
            "latency": champion_latency,
            "accuracy_ci_95": champion_accuracy_ci
        },
        
        # Challenger metrics
        "challenger": {
            "model_id": experiment.challenger_model_id,
            "performance": challenger_metrics,
            "latency": challenger_latency,
            "accuracy_ci_95": challenger_accuracy_ci
        },
        
        # Statistical comparison
        "statistical_tests": {
            "accuracy": {
                "test": "Two-proportion z-test",
                "difference": accuracy_diff,
                "difference_percent": accuracy_diff * 100,
                "p_value": accuracy_test["p_value"],
                "statistically_significant": accuracy_test["p_value"] < 0.05,
                "confidence_interval_95": accuracy_diff_ci,
                "effect_size_cohens_d": accuracy_effect_size
            },
            "latency": {
                "test": "Welch's t-test",
                "mean_difference_ms": challenger_latency["mean_ms"] - champion_latency["mean_ms"],
                "percent_change": latency_diff_pct,
                "p_value": latency_test["p_value"],
                "statistically_significant": latency_test["p_value"] < 0.05,
                "t_statistic": latency_test["t_statistic"],
                "degrees_of_freedom": latency_test["degrees_of_freedom"]
            }
        },
        
        # Recommendation
        "recommendation": {
            "decision": recommendation,
            "reasoning": reasoning,
            "confidence": "high" if min(accuracy_test["p_value"], latency_test["p_value"]) < 0.01 else "medium"
        }
    }
