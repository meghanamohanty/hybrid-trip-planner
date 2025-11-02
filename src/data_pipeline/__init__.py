"""
Data Pipeline Module
===================

Handles all data collection and preprocessing for the trip planner.
Fixed to avoid circular imports by using data_models.

Author: Hybrid Trip Planner Team
"""

# Version info
__version__ = "1.0.0"
__module_name__ = "data_pipeline"

# Import main data loading classes
from .osm_loader import OSMDataLoader
from .weather_loader import WeatherDataLoader  
from .data_validator import DataValidator

# Import data models
from .data_models import POI, WeatherData

# Import cache manager from utils (this should work now)
from ..utils.cache_manager import CacheManager

# Define public API
__all__ = [
    # Main classes
    "OSMDataLoader",
    "WeatherDataLoader", 
    "DataValidator",
    "CacheManager",
    
    # Data models
    "POI",
    "WeatherData"
]

def get_supported_data_sources():
    """Return list of supported data sources"""
    return {
        "osm": "OpenStreetMap via Overpass API",
        "weather": "WeatherAPI.com real-time weather data",
        "geocoding": "Built-in coordinate processing",
        "cache": "In-memory and file-based caching"
    }

def get_data_pipeline_info():
    """Return information about data pipeline capabilities"""
    return {
        "module": __module_name__,
        "version": __version__,
        "data_sources": get_supported_data_sources(),
        "features": [
            "POI collection from OpenStreetMap",
            "Weather-aware trip planning", 
            "Geospatial coordinate processing",
            "Data validation and cleaning",
            "Intelligent caching system"
        ]
    }

# Module-level configuration
DEFAULT_CACHE_TTL = 3600  # 1 hour
MAX_POI_RESULTS = 500
DEFAULT_SEARCH_RADIUS_KM = 10