"""
Configuration settings for the application
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    DATABASE_URL: str = "sqlite:///./ml_ab_testing.db"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Model Storage
    MODEL_STORAGE_PATH: str = "./models"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    
    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS string into list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    # Experiment Defaults
    DEFAULT_TRAFFIC_SPLIT_CHAMPION: int = 80
    DEFAULT_TRAFFIC_SPLIT_CHALLENGER: int = 20
    DEFAULT_MIN_SAMPLE_SIZE: int = 1000
    DEFAULT_SIGNIFICANCE_LEVEL: float = 0.05
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
