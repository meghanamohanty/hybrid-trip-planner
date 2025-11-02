"""
Configuration Management for Hybrid ML + GenAI Trip Planner
===========================================================

This module handles:
- Loading environment variables from .env file
- Validating required API keys and settings
- Providing centralized configuration access
- Setting up default values and error handling

Usage:
    from config import config
    api_key = config.GEMINI_API_KEY
    db_url = config.DATABASE_URL
"""

import os
import logging
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import validator


class Config(BaseSettings):
    """
    Centralized configuration class using Pydantic for validation
    Loads settings from environment variables with type checking
    """
    
    # =============================================================================
    # API KEYS - Required for core functionality
    # =============================================================================
    
    GEMINI_API_KEY: str
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    WEATHER_API_KEY: str
    
    @validator('GEMINI_API_KEY')
    def validate_gemini_key(cls, v):
        """Ensure Gemini API key is provided and properly formatted"""
        if not v or len(v) < 20:
            raise ValueError("Valid Gemini API key is required")
        return v
    
    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    
    DATABASE_URL: str
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_KEY: str
    
    @validator('DATABASE_URL')
    def validate_database_url(cls, v):
        """Ensure database URL is properly formatted"""
        if not v.startswith('postgresql://'):
            raise ValueError("Database URL must be a valid PostgreSQL connection string")
        return v
    
    # =============================================================================
    # CACHE CONFIGURATION
    # =============================================================================
    
    CACHE_TYPE: str = "memory"
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds
    CACHE_MAX_SIZE: int = 1000  # Maximum cache entries
    OS_CACHE_DIR: str = "./data/cache"
    
    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    
    APP_NAME: str = "Hybrid Trip Planner"
    APP_VERSION: str = "1.0.0"
    DEBUG_MODE: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Default user preferences
    DEFAULT_BUDGET_CURRENCY: str = "INR"
    DEFAULT_TRAVEL_MODE: str = "walking"
    DEFAULT_TRIP_DAYS: int = 3
    
    # =============================================================================
    # API TIMEOUTS AND RATE LIMITS
    # =============================================================================
    
    OSM_API_TIMEOUT: int = 30
    OSM_RATE_LIMIT_DELAY: int = 1
    METEOSTAT_TIMEOUT: int = 15
    GOOGLE_API_TIMEOUT: int = 10
    GOOGLE_API_RETRY_COUNT: int = 3
    
    # =============================================================================
    # ML ALGORITHM PARAMETERS
    # =============================================================================
    
    # POI Scoring weights (must sum to 1.0)
    POPULARITY_WEIGHT: float = 0.35
    INTEREST_WEIGHT: float = 0.25
    WEATHER_WEIGHT: float = 0.15
    DISTANCE_WEIGHT: float = 0.15
    PRICE_WEIGHT: float = 0.10
    
    @validator('POPULARITY_WEIGHT', 'INTEREST_WEIGHT', 'WEATHER_WEIGHT', 
              'DISTANCE_WEIGHT', 'PRICE_WEIGHT', always=True)
    def validate_weights(cls, v, values):
        """Ensure all weights are between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {v}")
        return v
    
    # Clustering parameters
    MAX_CLUSTER_RADIUS_KM: int = 5
    MIN_POIS_PER_DAY: int = 3
    MAX_POIS_PER_DAY: int = 8
    
    # Budget distribution percentages
    HOTEL_BUDGET_PERCENT: int = 40
    FOOD_BUDGET_PERCENT: int = 25
    TRANSPORT_BUDGET_PERCENT: int = 15
    TICKETS_BUDGET_PERCENT: int = 20
    
    @validator('HOTEL_BUDGET_PERCENT', 'FOOD_BUDGET_PERCENT', 
              'TRANSPORT_BUDGET_PERCENT', 'TICKETS_BUDGET_PERCENT', always=True)
    def validate_budget_distribution(cls, v, values):
        """Ensure budget percentages sum to 100"""
        # This is a simplified check - in practice you'd sum all values
        if not 0 <= v <= 100:
            raise ValueError(f"Budget percentage must be between 0 and 100, got {v}")
        return v
    
    # =============================================================================
    # FILE PATHS
    # =============================================================================
    
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    CACHE_DATA_DIR: str = "./data/cache"
    OUTPUT_DIR: str = "./outputs"
    LOG_FILE_PATH: str = "./logs/app.log"
    ERROR_LOG_PATH: str = "./logs/errors.log"
    
    # =============================================================================
    # DEVELOPMENT SETTINGS
    # =============================================================================
    
    DEVELOPMENT: bool = True
    ENABLE_DEBUG_LOGGING: bool = True
    STREAMLIT_SERVER_PORT: int = 8501
    FASTAPI_SERVER_PORT: int = 8000
    
    # Testing
    ENABLE_TESTING_MODE: bool = False
    TEST_DATA_SIZE_LIMIT: int = 100
    
    # =============================================================================
    # CONFIGURATION SETUP
    # =============================================================================
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR, 
            self.CACHE_DATA_DIR,
            self.OUTPUT_DIR,
            os.path.dirname(self.LOG_FILE_PATH),
            os.path.dirname(self.ERROR_LOG_PATH)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self) -> None:
        """Configure application logging"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(self.LOG_FILE_PATH),
                logging.StreamHandler()  # Console output
            ]
        )
        
        # Create error-specific logger
        error_handler = logging.FileHandler(self.ERROR_LOG_PATH)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(log_format))
        
        logger = logging.getLogger()
        logger.addHandler(error_handler)
    
    def validate_configuration(self) -> bool:
        """
        Validate that all critical configuration is properly set
        Returns True if configuration is valid, raises exception otherwise
        """
        try:
            # Check if weights sum to approximately 1.0
            weight_sum = (self.POPULARITY_WEIGHT + self.INTEREST_WEIGHT + 
                         self.WEATHER_WEIGHT + self.DISTANCE_WEIGHT + self.PRICE_WEIGHT)
            
            if not 0.95 <= weight_sum <= 1.05:  # Allow small floating point errors
                raise ValueError(f"POI scoring weights must sum to 1.0, got {weight_sum}")
            
            # Check if budget percentages sum to 100
            budget_sum = (self.HOTEL_BUDGET_PERCENT + self.FOOD_BUDGET_PERCENT + 
                         self.TRANSPORT_BUDGET_PERCENT + self.TICKETS_BUDGET_PERCENT)
            
            if budget_sum != 100:
                raise ValueError(f"Budget percentages must sum to 100, got {budget_sum}")
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            raise


def load_configuration() -> Config:
    """
    Load and validate configuration from environment
    Creates directories and sets up logging
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Create and validate configuration
        config = Config()
        
        # Create necessary directories
        config.create_directories()
        
        # Setup logging
        config.setup_logging()
        
        # Validate configuration
        config.validate_configuration()
        
        logging.info(f"Configuration loaded successfully for {config.APP_NAME} v{config.APP_VERSION}")
        return config
        
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        raise


# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# =============================================================================

# Create global configuration instance
config = load_configuration()

# Export commonly used settings for easy access
GEMINI_API_KEY = config.GEMINI_API_KEY
DATABASE_URL = config.DATABASE_URL
DEBUG_MODE = config.DEBUG_MODE