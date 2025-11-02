"""
Data Models for Trip Planner
============================

Centralized data models to avoid circular imports.
Contains POI, WeatherData, and other shared data structures.

Author: Hybrid Trip Planner Team
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import date


@dataclass
class POI:
    """
    Data class representing a Point of Interest from OpenStreetMap
    
    Attributes:
        osm_id (str): Unique OpenStreetMap identifier
        name (str): POI name
        category (str): POI category (tourism, restaurant, hotel, etc.)
        subcategory (str): Specific type (museum, cafe, park, etc.)
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        address (str): Street address if available
        phone (str): Phone number if available
        website (str): Website URL if available
        opening_hours (str): Opening hours if available
        rating (float): User rating if available (0-5 scale)
        price_level (int): Price level indicator (1-4, 1=cheap, 4=expensive)
        tags (Dict): Raw OSM tags for additional information
        outdoor (bool): Whether POI is primarily outdoor activity
        wheelchair_accessible (bool): Wheelchair accessibility
        fee_required (bool): Whether entry fee is required
    """
    osm_id: str
    name: str
    category: str
    subcategory: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    opening_hours: Optional[str] = None
    rating: Optional[float] = None
    price_level: Optional[int] = None
    tags: Optional[Dict] = None
    outdoor: Optional[bool] = None
    wheelchair_accessible: Optional[bool] = None
    fee_required: Optional[bool] = None


@dataclass
class WeatherData:
    """
    Data class representing daily weather information from WeatherAPI.com
    
    Attributes:
        date (date): Date of weather data
        location_name (str): Name of the location
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        temperature_avg (float): Average temperature in Celsius
        temperature_min (float): Minimum temperature in Celsius
        temperature_max (float): Maximum temperature in Celsius
        precipitation (float): Precipitation in mm
        humidity (float): Average relative humidity percentage
        wind_speed (float): Wind speed in km/h
        pressure (float): Atmospheric pressure in hPa
        weather_condition (str): General weather condition description
        weather_icon (str): Weather condition icon code
        uv_index (float): UV index for sun exposure planning
        visibility (float): Visibility in km
        cloud_cover (int): Cloud cover percentage
        feels_like (float): Feels like temperature in Celsius
        sunrise (str): Sunrise time (HH:MM format)
        sunset (str): Sunset time (HH:MM format)
        moon_phase (str): Moon phase description
        outdoor_suitability (float): Suitability score for outdoor activities (0-1)
        indoor_recommendation (bool): Whether indoor activities are recommended
        weather_penalty (float): Penalty score for outdoor POIs (0-1)
        weather_note (str): Human-readable weather note for itinerary
    """
    date: date
    location_name: str
    latitude: float
    longitude: float
    temperature_avg: Optional[float] = None
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    precipitation: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    pressure: Optional[float] = None
    weather_condition: Optional[str] = None
    weather_icon: Optional[str] = None
    uv_index: Optional[float] = None
    visibility: Optional[float] = None
    cloud_cover: Optional[int] = None
    feels_like: Optional[float] = None
    sunrise: Optional[str] = None
    sunset: Optional[str] = None
    moon_phase: Optional[str] = None
    outdoor_suitability: Optional[float] = None
    indoor_recommendation: Optional[bool] = None
    weather_penalty: Optional[float] = None
    weather_note: Optional[str] = None


# Export classes for easy import
__all__ = ['POI', 'WeatherData']