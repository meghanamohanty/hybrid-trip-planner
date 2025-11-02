"""
Utilities Module
===============

Common utility functions and helpers used across all modules:
- Logging configuration and management
- Caching mechanisms (in-memory and file-based)
- Data validation and cleaning utilities
- Error handling and exception management
- Performance monitoring and profiling tools
- Common data processing functions

Classes:
    Logger: Centralized logging management with different log levels
    CacheManager: Handles in-memory and persistent caching strategies
    ErrorHandler: Standardized error handling and exception management
    PerformanceMonitor: Tracks execution time and system performance
    ConfigHelper: Configuration management and environment utilities

Functions:
    setup_logging(): Initialize application-wide logging
    validate_coordinates(): Check if latitude/longitude are valid
    clean_text(): Clean and normalize text data
    format_currency(): Format monetary values according to locale
    calculate_distance(): Calculate distance between coordinates
    handle_api_error(): Standardized API error handling
    measure_execution_time(): Performance timing decorator
"""

# Version and module info
__version__ = "1.0.0"
__module_name__ = "utils"

# Import core utility classes (REMOVED DataValidator to fix circular import)
from .cache_manager import CacheManager
from .error_handler import ErrorHandler
from .performance_monitor import PerformanceMonitor

# Import data processing utilities
from .data_utils import (
    clean_text,
    validate_coordinates,
    calculate_distance,
    normalize_data,
    parse_json_safely,
    convert_currency
)

# Define public API (REMOVED DataValidator from here)
__all__ = [
    # Core utility classes
    "CacheManager", 
    "ErrorHandler",
    "PerformanceMonitor",
    
    # Data processing functions
    "clean_text",
    "validate_coordinates",
    "calculate_distance",
    "normalize_data",
    "parse_json_safely",
    "convert_currency",
]

def get_common_constants():
    """
    Return commonly used constants across the application
    """
    return {
        # Geographic constants
        "EARTH_RADIUS_KM": 6371,
        "MAX_LATITUDE": 90.0,
        "MIN_LATITUDE": -90.0,
        "MAX_LONGITUDE": 180.0,
        "MIN_LONGITUDE": -180.0,
        
        # Time constants
        "SECONDS_PER_MINUTE": 60,
        "MINUTES_PER_HOUR": 60,
        "HOURS_PER_DAY": 24,
        
        # Currency constants
        "DEFAULT_CURRENCY": "INR",
        "SUPPORTED_CURRENCIES": ["INR", "USD", "EUR", "GBP"],
        
        # API constants
        "DEFAULT_TIMEOUT": 30,
        "MAX_RETRIES": 3,
        "RATE_LIMIT_DELAY": 1,
        
        # Data validation
        "MAX_STRING_LENGTH": 1000,
        "MIN_PASSWORD_LENGTH": 8,
        "MAX_FILE_SIZE_MB": 10
    }

def get_logging_config():
    """
    Return logging configuration for different environments
    """
    return {
        "development": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": ["console", "file"]
        },
        "production": {
            "level": "INFO", 
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "handlers": ["file", "syslog"]
        },
        "testing": {
            "level": "WARNING",
            "format": "%(levelname)s - %(message)s", 
            "handlers": ["console"]
        }
    }

def get_cache_strategies():
    """
    Return available caching strategies and their configurations
    """
    return {
        "memory": {
            "type": "in_memory",
            "max_size": 1000,
            "ttl_seconds": 3600,
            "eviction_policy": "LRU"
        },
        "file": {
            "type": "file_based",
            "cache_dir": "./data/cache",
            "max_file_size_mb": 100,
            "cleanup_interval_hours": 24
        },
        "hybrid": {
            "type": "memory_and_file",
            "memory_max_size": 500,
            "file_fallback": True,
            "sync_interval_minutes": 30
        }
    }

def get_error_codes():
    """
    Return standardized error codes for the application
    """
    return {
        # Data errors (1000-1999)
        "DATA_VALIDATION_ERROR": 1001,
        "DATA_NOT_FOUND": 1002,
        "DATA_CORRUPTION": 1003,
        "INVALID_COORDINATES": 1004,
        
        # API errors (2000-2999)
        "API_RATE_LIMIT": 2001,
        "API_TIMEOUT": 2002,
        "API_AUTHENTICATION": 2003,
        "API_SERVICE_UNAVAILABLE": 2004,
        
        # Configuration errors (3000-3999)
        "CONFIG_MISSING": 3001,
        "CONFIG_INVALID": 3002,
        "ENV_VAR_MISSING": 3003,
        
        # Processing errors (4000-4999)
        "CLUSTERING_FAILED": 4001,
        "ROUTING_FAILED": 4002,
        "AI_GENERATION_FAILED": 4003,
        "BUDGET_CALCULATION_ERROR": 4004,
        
        # UI errors (5000-5999)
        "INVALID_USER_INPUT": 5001,
        "EXPORT_FAILED": 5002,
        "VISUALIZATION_ERROR": 5003
    }

def get_utils_info():
    """
    Return comprehensive information about utils module capabilities
    """
    return {
        "module": __module_name__,
        "version": __version__,
        "features": [
            "Centralized logging with multiple handlers and levels",
            "Flexible caching strategies (memory, file, hybrid)",
            "Comprehensive data validation and cleaning",
            "Robust error handling with standardized codes",
            "Performance monitoring and profiling tools",
            "API utilities with retry and rate limiting",
            "File I/O helpers with safety checks",
            "Time and date processing utilities",
            "Geographic calculations and validations"
        ],
        "utility_categories": [
            "Data Processing & Validation",
            "API & Network Operations", 
            "File & I/O Management",
            "Time & Date Handling",
            "Caching & Performance",
            "Logging & Error Handling"
        ],
        "constants": list(get_common_constants().keys()),
        "cache_strategies": list(get_cache_strategies().keys()),
        "error_codes": len(get_error_codes()),
        "logging_levels": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    }

# Module-level decorators for common operations
def measure_time(func):
    """Decorator to measure function execution time"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # logger = Logger()
        # logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def cache_result(ttl_seconds=3600):
    """Decorator to cache function results"""
    def decorator(func):
        import functools
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = str(args) + str(kwargs)
            if cache_key in cache:
                return cache[cache_key]
            
            result = func(*args, **kwargs)
            cache[cache_key] = result
            return result
        return wrapper
    return decorator

# Export commonly used constants
EARTH_RADIUS_KM = 6371
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
DEFAULT_CACHE_TTL = 3600