"""
Data Validation and Processing Utilities
=======================================

Essential data validation and processing functions used across the application.
Provides coordinate validation, distance calculations, text processing, and data formatting.

Key Features:
- Geographic coordinate validation and calculations
- Haversine distance formula for coordinate pairs
- Text cleaning and normalization from API responses
- Data type validation and conversion utilities
- JSON parsing with error handling
- Currency and numeric formatting

Functions:
    validate_coordinates: Check if latitude/longitude are valid
    calculate_distance: Calculate distance between two coordinate pairs
    clean_text: Clean and normalize text data
    normalize_data: Normalize data for ML processing
    parse_json_safely: Safe JSON parsing with error handling
    format_currency: Format monetary values with locale support

Author: Hybrid Trip Planner Team
"""

import re
import json
import math
import logging
from typing import Any, Optional, Dict, List, Tuple, Union
from datetime import datetime, date
import unicodedata

# Import configuration
from config import config


# Module logger
logger = logging.getLogger(__name__)

# Constants for geographic calculations
EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers
MAX_LATITUDE = 90.0
MIN_LATITUDE = -90.0
MAX_LONGITUDE = 180.0
MIN_LONGITUDE = -180.0


def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Validate if latitude and longitude coordinates are within valid ranges
    
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        
    Returns:
        bool: True if coordinates are valid, False otherwise
        
    Examples:
        >>> validate_coordinates(17.6868, 83.2185)  # Visakhapatnam
        True
        >>> validate_coordinates(91.0, 181.0)  # Invalid
        False
    """
    try:
        # Check if values are numeric
        lat = float(latitude)
        lon = float(longitude)
        
        # Check latitude range
        if not (MIN_LATITUDE <= lat <= MAX_LATITUDE):
            return False
        
        # Check longitude range
        if not (MIN_LONGITUDE <= lon <= MAX_LONGITUDE):
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula
    
    Args:
        lat1 (float): Latitude of first point
        lon1 (float): Longitude of first point
        lat2 (float): Latitude of second point
        lon2 (float): Longitude of second point
        
    Returns:
        float: Distance in kilometers, None if coordinates are invalid
        
    Examples:
        >>> calculate_distance(17.6868, 83.2185, 17.7231, 83.3210)  # Within Visakhapatnam
        8.42
    """
    try:
        # Validate coordinates
        if not (validate_coordinates(lat1, lon1) and validate_coordinates(lat2, lon2)):
            return None
        
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Distance in kilometers
        distance = EARTH_RADIUS_KM * c
        
        return round(distance, 2)
        
    except (ValueError, TypeError, OverflowError):
        logger.warning(f"Error calculating distance between ({lat1}, {lon1}) and ({lat2}, {lon2})")
        return None


def clean_text(text: str, remove_html: bool = True, normalize_unicode: bool = True,
               max_length: int = None) -> Optional[str]:
    """
    Clean and normalize text data from API responses
    
    Args:
        text (str): Text to clean
        remove_html (bool): Whether to remove HTML tags
        normalize_unicode (bool): Whether to normalize Unicode characters
        max_length (int): Maximum length to truncate to
        
    Returns:
        str: Cleaned text, None if input is invalid
        
    Examples:
        >>> clean_text("<p>Kailasagiri Park</p>")
        "Kailasagiri Park"
        >>> clean_text("Café Résumé", normalize_unicode=True)
        "Cafe Resume"
    """
    try:
        if not text or not isinstance(text, str):
            return None
        
        # Remove HTML tags if requested
        if remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize Unicode characters if requested
        if normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
            text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate if max_length specified
        if max_length and len(text) > max_length:
            text = text[:max_length].strip()
            if text.endswith('...'):
                text = text[:-3].strip() + '...'
            else:
                text = text + '...'
        
        return text.strip() if text.strip() else None
        
    except Exception as e:
        logger.warning(f"Error cleaning text '{text[:50]}...': {e}")
        return None


def normalize_data(data: Any, data_type: str = "auto") -> Any:
    """
    Normalize data for consistent processing
    
    Args:
        data (Any): Data to normalize
        data_type (str): Expected data type ("numeric", "text", "auto")
        
    Returns:
        Any: Normalized data
        
    Examples:
        >>> normalize_data("  123.45  ", "numeric")
        123.45
        >>> normalize_data("RESTAURANT", "text")
        "restaurant"
    """
    try:
        if data is None:
            return None
        
        # Auto-detect data type if not specified
        if data_type == "auto":
            if isinstance(data, (int, float)):
                data_type = "numeric"
            elif isinstance(data, str):
                # Try to detect if it's a number
                cleaned = str(data).strip()
                try:
                    float(cleaned)
                    data_type = "numeric"
                except ValueError:
                    data_type = "text"
            else:
                return data
        
        # Normalize based on type
        if data_type == "numeric":
            if isinstance(data, (int, float)):
                return data
            elif isinstance(data, str):
                cleaned = data.strip().replace(',', '')
                try:
                    # Try integer first
                    if '.' not in cleaned:
                        return int(cleaned)
                    else:
                        return float(cleaned)
                except ValueError:
                    return None
        
        elif data_type == "text":
            if isinstance(data, str):
                return clean_text(data)
            else:
                return clean_text(str(data))
        
        return data
        
    except Exception as e:
        logger.warning(f"Error normalizing data '{data}': {e}")
        return data


def parse_json_safely(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with error handling
    
    Args:
        json_string (str): JSON string to parse
        default (Any): Default value to return on error
        
    Returns:
        Any: Parsed JSON data or default value
        
    Examples:
        >>> parse_json_safely('{"name": "test"}')
        {"name": "test"}
        >>> parse_json_safely('invalid json', {})
        {}
    """
    try:
        if not json_string or not isinstance(json_string, str):
            return default
        
        return json.loads(json_string.strip())
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"JSON parsing failed: {e}")
        return default


def convert_currency(amount: float, from_currency: str = "INR", 
                    to_currency: str = "INR") -> float:
    """
    Convert currency amounts (basic implementation)
    
    Args:
        amount (float): Amount to convert
        from_currency (str): Source currency code
        to_currency (str): Target currency code
        
    Returns:
        float: Converted amount
        
    Note:
        This is a basic implementation. In production, use real exchange rates.
    """
    try:
        # Basic exchange rates (hardcoded for MVP)
        exchange_rates = {
            "INR": 1.0,
            "USD": 83.0,  # 1 USD = 83 INR (approximate)
            "EUR": 90.0,  # 1 EUR = 90 INR (approximate)
            "GBP": 105.0  # 1 GBP = 105 INR (approximate)
        }
        
        if from_currency == to_currency:
            return amount
        
        # Convert to INR first, then to target currency
        inr_amount = amount * exchange_rates.get(from_currency, 1.0)
        converted_amount = inr_amount / exchange_rates.get(to_currency, 1.0)
        
        return round(converted_amount, 2)
        
    except Exception as e:
        logger.warning(f"Currency conversion error: {e}")
        return amount


def format_currency(amount: float, currency: str = "INR", 
                   include_symbol: bool = True) -> str:
    """
    Format currency amount for display
    
    Args:
        amount (float): Amount to format
        currency (str): Currency code
        include_symbol (bool): Whether to include currency symbol
        
    Returns:
        str: Formatted currency string
        
    Examples:
        >>> format_currency(1500.50, "INR")
        "₹1,500.50"
        >>> format_currency(25.99, "USD")
        "$25.99"
    """
    try:
        # Currency symbols
        symbols = {
            "INR": "₹",
            "USD": "$",
            "EUR": "€",
            "GBP": "£"
        }
        
        # Format number with commas
        if amount >= 1000:
            formatted = f"{amount:,.2f}"
        else:
            formatted = f"{amount:.2f}"
        
        # Add currency symbol if requested
        if include_symbol and currency in symbols:
            return f"{symbols[currency]}{formatted}"
        else:
            return f"{formatted} {currency}"
            
    except Exception as e:
        logger.warning(f"Currency formatting error: {e}")
        return str(amount)


def validate_budget_range(budget: float, min_budget: float = 1000, 
                         max_budget: float = 500000) -> bool:
    """
    Validate if budget is within acceptable range
    
    Args:
        budget (float): Budget amount to validate
        min_budget (float): Minimum acceptable budget
        max_budget (float): Maximum acceptable budget
        
    Returns:
        bool: True if budget is valid
        
    Examples:
        >>> validate_budget_range(15000)
        True
        >>> validate_budget_range(100)  # Too low
        False
    """
    try:
        budget_val = float(budget)
        return min_budget <= budget_val <= max_budget
    except (ValueError, TypeError):
        return False


def validate_coordinates_range(coordinates: List[Tuple[float, float]], 
                              max_radius_km: float = 50) -> bool:
    """
    Validate if a list of coordinates are within reasonable range of each other
    
    Args:
        coordinates (List[Tuple[float, float]]): List of (lat, lon) tuples
        max_radius_km (float): Maximum radius in kilometers
        
    Returns:
        bool: True if all coordinates are within range
    """
    try:
        if len(coordinates) < 2:
            return True
        
        # Find center point (simple average)
        center_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
        center_lon = sum(coord[1] for coord in coordinates) / len(coordinates)
        
        # Check if all points are within max_radius_km of center
        for lat, lon in coordinates:
            if not validate_coordinates(lat, lon):
                return False
            
            distance = calculate_distance(center_lat, center_lon, lat, lon)
            if distance is None or distance > max_radius_km:
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Coordinate range validation error: {e}")
        return False


def sanitize_input(input_data: Any, max_length: int = 1000, 
                  allowed_chars: str = None) -> Optional[str]:
    """
    Sanitize user input for security and consistency
    
    Args:
        input_data (Any): Input data to sanitize
        max_length (int): Maximum allowed length
        allowed_chars (str): Regex pattern for allowed characters
        
    Returns:
        str: Sanitized input, None if invalid
        
    Examples:
        >>> sanitize_input("Visakhapatnam<script>", max_length=50)
        "Visakhapatnam"
    """
    try:
        if input_data is None:
            return None
        
        # Convert to string
        text = str(input_data).strip()
        
        # Remove HTML/script tags for security
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\';]', '', text)
        
        # Apply allowed characters filter if specified
        if allowed_chars:
            text = re.sub(f'[^{allowed_chars}]', '', text)
        
        # Truncate to max length
        if len(text) > max_length:
            text = text[:max_length]
        
        return text if text else None
        
    except Exception as e:
        logger.warning(f"Input sanitization error: {e}")
        return None


def get_bounding_box(coordinates: List[Tuple[float, float]], 
                    padding_km: float = 1.0) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate bounding box for a list of coordinates with padding
    
    Args:
        coordinates (List[Tuple[float, float]]): List of (lat, lon) coordinates
        padding_km (float): Padding distance in kilometers
        
    Returns:
        Tuple[float, float, float, float]: (min_lat, min_lon, max_lat, max_lon)
        None if coordinates are invalid
    """
    try:
        if not coordinates:
            return None
        
        # Validate all coordinates
        valid_coords = [(lat, lon) for lat, lon in coordinates 
                       if validate_coordinates(lat, lon)]
        
        if not valid_coords:
            return None
        
        # Find min/max coordinates
        latitudes = [coord[0] for coord in valid_coords]
        longitudes = [coord[1] for coord in valid_coords]
        
        min_lat = min(latitudes)
        max_lat = max(latitudes)
        min_lon = min(longitudes)
        max_lon = max(longitudes)
        
        # Add padding (approximate conversion: 1 degree ≈ 111 km)
        padding_degrees = padding_km / 111.0
        
        return (
            max(min_lat - padding_degrees, MIN_LATITUDE),
            max(min_lon - padding_degrees, MIN_LONGITUDE),
            min(max_lat + padding_degrees, MAX_LATITUDE),
            min(max_lon + padding_degrees, MAX_LONGITUDE)
        )
        
    except Exception as e:
        logger.warning(f"Bounding box calculation error: {e}")
        return None


def format_datetime(dt: Union[datetime, date, str], 
                   format_string: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
    """
    Format datetime object for consistent display
    
    Args:
        dt (Union[datetime, date, str]): Datetime to format
        format_string (str): Format string
        
    Returns:
        str: Formatted datetime string, None if invalid
    """
    try:
        if isinstance(dt, str):
            # Try to parse string datetime
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        elif isinstance(dt, date) and not isinstance(dt, datetime):
            # Convert date to datetime
            dt = datetime.combine(dt, datetime.min.time())
        
        if isinstance(dt, datetime):
            return dt.strftime(format_string)
        
        return None
        
    except Exception as e:
        logger.warning(f"Datetime formatting error: {e}")
        return None


def get_data_quality_score(data: Dict, required_fields: List[str], 
                          optional_fields: List[str] = None) -> float:
    """
    Calculate data quality score based on field completeness
    
    Args:
        data (Dict): Data to evaluate
        required_fields (List[str]): Required field names
        optional_fields (List[str]): Optional field names
        
    Returns:
        float: Quality score (0-100)
    """
    try:
        if not data or not required_fields:
            return 0.0
        
        score = 0.0
        
        # Required fields (70% of score)
        required_filled = sum(1 for field in required_fields 
                            if field in data and data[field] is not None)
        required_score = (required_filled / len(required_fields)) * 70
        score += required_score
        
        # Optional fields (30% of score)
        if optional_fields:
            optional_filled = sum(1 for field in optional_fields 
                                if field in data and data[field] is not None)
            optional_score = (optional_filled / len(optional_fields)) * 30
            score += optional_score
        else:
            score += 30  # Full optional score if no optional fields defined
        
        return round(score, 2)
        
    except Exception as e:
        logger.warning(f"Quality score calculation error: {e}")
        return 0.0