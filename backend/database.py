"""
Database models and initialization
"""
from sqlalchemy import create_engine, Column, String, Integer, Float, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

from backend.config import settings

# Create base class for models
Base = declarative_base()

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Model(Base):
    """ML Model registry table"""
    __tablename__ = "models"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    file_path = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)  # sklearn, xgboost, tensorflow
    alias = Column(String(50))  # champion, challenger
    registered_at = Column(DateTime, default=datetime.utcnow)
    model_metadata = Column(JSON)  # Changed from 'metadata' to avoid SQLAlchemy reserved word
    
    # Relationships
    champion_experiments = relationship("Experiment", foreign_keys="Experiment.champion_model_id", back_populates="champion_model")
    challenger_experiments = relationship("Experiment", foreign_keys="Experiment.challenger_model_id", back_populates="challenger_model")


class Experiment(Base):
    """A/B Test Experiment table"""
    __tablename__ = "experiments"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    champion_model_id = Column(String(50), ForeignKey("models.id"), nullable=False)
    challenger_model_id = Column(String(50), ForeignKey("models.id"), nullable=False)
    traffic_split = Column(JSON, nullable=False)  # {"champion": 80, "challenger": 20}
    primary_metric = Column(String(50), nullable=False)
    secondary_metrics = Column(JSON)
    status = Column(String(20), default="draft")  # draft, running, completed, stopped
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    min_sample_size = Column(Integer, default=1000)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    champion_model = relationship("Model", foreign_keys=[champion_model_id], back_populates="champion_experiments")
    challenger_model = relationship("Model", foreign_keys=[challenger_model_id], back_populates="challenger_experiments")
    predictions = relationship("Prediction", back_populates="experiment")


class Prediction(Base):
    """Prediction logs table"""
    __tablename__ = "predictions"
    
    prediction_id = Column(String(50), primary_key=True)
    experiment_id = Column(String(50), ForeignKey("experiments.id"), nullable=False)
    user_id = Column(String(100), nullable=False)
    model_id = Column(String(50), ForeignKey("models.id"), nullable=False)
    variant = Column(String(20), nullable=False)  # champion or challenger
    features = Column(JSON, nullable=False)
    prediction = Column(Integer)  # For classification
    prediction_value = Column(Float)  # For regression or probability
    probability = Column(Float)
    ground_truth = Column(Integer)
    ground_truth_value = Column(Float)
    latency_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    ground_truth_timestamp = Column(DateTime)
    prediction_metadata = Column(JSON)  # Changed from 'metadata' to avoid SQLAlchemy reserved word
    
    # Relationships
    experiment = relationship("Experiment", back_populates="predictions")


def init_db():
    """Initialize database - create all tables"""
    # Create models directory if it doesn't exist
    os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
