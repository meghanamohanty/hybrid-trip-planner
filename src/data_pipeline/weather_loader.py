"""
WeatherAPI.com Weather Data Loader
=================================

Fetches weather data from WeatherAPI.com for trip planning.
Fixed to use WeatherData from data_models to avoid circular imports.

Author: Hybrid Trip Planner Team
"""

import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple
import requests
import json

# Import WeatherData from data_models (not from this module)
from .data_models import WeatherData

# Import configuration
from config import config


class WeatherAnalyzer:
    """Helper class for analyzing weather conditions and calculating penalties"""
    
    @staticmethod
    def calculate_outdoor_suitability(temperature: float, precipitation: float, 
                                    wind_speed: float = 0) -> float:
        """Calculate suitability score for outdoor activities"""
        score = 1.0
        
        # Temperature penalties
        if temperature < 10:  # Too cold
            score -= 0.4
        elif temperature < 18:  # Cool
            score -= 0.2
        elif temperature > 35:  # Too hot
            score -= 0.3
        elif temperature > 30:  # Hot
            score -= 0.1
        
        # Precipitation penalties
        if precipitation > 10:  # Heavy rain
            score -= 0.6
        elif precipitation > 5:  # Moderate rain
            score -= 0.4
        elif precipitation > 1:  # Light rain
            score -= 0.2
        
        # Wind penalties
        if wind_speed > 25:  # Strong wind
            score -= 0.3
        elif wind_speed > 15:  # Moderate wind
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def should_recommend_indoor(temperature: float, precipitation: float, 
                              wind_speed: float = 0) -> bool:
        """Determine if indoor activities should be recommended"""
        # Heavy rain or extreme temperatures
        if precipitation > 5 or temperature < 10 or temperature > 35:
            return True
        
        # Strong wind
        if wind_speed > 20:
            return True
        
        return False
    
    @staticmethod
    def calculate_weather_penalty(temperature: float, precipitation: float, 
                                wind_speed: float = 0) -> float:
        """Calculate weather penalty for outdoor POI scoring"""
        return 1.0 - WeatherAnalyzer.calculate_outdoor_suitability(
            temperature, precipitation, wind_speed
        )
    
    @staticmethod
    def generate_weather_note(weather_data: WeatherData) -> str:
        """Generate human-readable weather note for itinerary"""
        temp = weather_data.temperature_avg or 0
        precip = weather_data.precipitation or 0
        
        note_parts = []
        
        # Temperature note
        if temp < 15:
            note_parts.append("Dress warmly")
        elif temp > 30:
            note_parts.append("Stay hydrated")
        
        # Rain note
        if precip > 5:
            note_parts.append("Carry umbrella")
            note_parts.append("Consider indoor activities")
        elif precip > 1:
            note_parts.append("Light rain possible")
        
        # Activity recommendations
        if weather_data.indoor_recommendation:
            note_parts.append("Indoor activities recommended")
        elif weather_data.outdoor_suitability and weather_data.outdoor_suitability > 0.7:
            note_parts.append("Great weather for outdoor activities")
        
        return "; ".join(note_parts) if note_parts else "Good weather for sightseeing"


class WeatherDataLoader:
    """Main class for loading weather data from WeatherAPI.com"""
    
    def __init__(self):
        """Initialize Weather Data Loader with WeatherAPI.com configuration"""
        self.logger = logging.getLogger(__name__)
        self.analyzer = WeatherAnalyzer()
        
        # WeatherAPI.com configuration
        self.api_key = config.WEATHER_API_KEY  # Fixed config name
        self.base_url = "http://api.weatherapi.com/v1"
        self.timeout = config.METEOSTAT_TIMEOUT
        self.max_retries = 3
        self.rate_limit_delay = 1
        
        # API endpoints
        self.current_endpoint = f"{self.base_url}/current.json"
        self.forecast_endpoint = f"{self.base_url}/forecast.json"
        self.history_endpoint = f"{self.base_url}/history.json"
        
        self.logger.info("WeatherAPI.com Data Loader initialized")
    
    def get_weather_for_dates(self, latitude: float, longitude: float, 
                            dates: List[date], location_name: str = "") -> List[WeatherData]:
        """Get weather data for specific dates at a location"""
        # Validate coordinates
        if not self._validate_coordinates(latitude, longitude):
            self.logger.error(f"Invalid coordinates: {latitude}, {longitude}")
            return []
        
        try:
            weather_data_list = []
            location_query = f"{latitude},{longitude}"
            
            # Group dates by time period for efficient API calls
            today = date.today()
            historical_dates = [d for d in dates if d < today]
            current_date = [d for d in dates if d == today]
            future_dates = [d for d in dates if d > today]
            
            # Fetch historical data (up to 365 days ago)
            if historical_dates:
                historical_data = self._fetch_historical_weather(
                    location_query, historical_dates, location_name
                )
                weather_data_list.extend(historical_data)
            
            # Fetch current weather
            if current_date:
                current_data = self._fetch_current_weather(
                    location_query, location_name
                )
                if current_data:
                    weather_data_list.append(current_data)
            
            # Fetch forecast data (up to 14 days ahead)
            if future_dates:
                forecast_data = self._fetch_forecast_weather(
                    location_query, future_dates, location_name
                )
                weather_data_list.extend(forecast_data)
            
            # Sort by date
            weather_data_list.sort(key=lambda x: x.date)
            
            self.logger.info(f"Retrieved weather data for {len(weather_data_list)} days at {location_name}")
            return weather_data_list
            
        except Exception as e:
            self.logger.error(f"Error fetching weather data for {location_name}: {e}")
            return []
    
    def get_weather_for_trip(self, latitude: float, longitude: float, 
                           start_date: date, num_days: int, 
                           location_name: str = "") -> List[WeatherData]:
        """Get weather data for a trip duration"""
        # Generate date list
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        
        return self.get_weather_for_dates(latitude, longitude, dates, location_name)
    
    def _validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Validate coordinates are within valid ranges"""
        try:
            lat = float(latitude)
            lon = float(longitude)
            return -90 <= lat <= 90 and -180 <= lon <= 180
        except (ValueError, TypeError):
            return False
    
    def _fetch_historical_weather(self, location_query: str, dates: List[date], 
                                location_name: str) -> List[WeatherData]:
        """Fetch historical weather data using WeatherAPI.com"""
        weather_list = []
        
        for target_date in dates:
            try:
                # Check if date is within API limits (365 days ago)
                days_ago = (date.today() - target_date).days
                if days_ago > 365:
                    self.logger.warning(f"Date {target_date} is beyond API limit (365 days)")
                    continue
                
                # API request for historical data
                params = {
                    "key": self.api_key,
                    "q": location_query,
                    "dt": target_date.strftime("%Y-%m-%d")
                }
                
                response = requests.get(
                    self.history_endpoint, 
                    params=params, 
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    weather_data = self._parse_weather_response(
                        data, target_date, location_name
                    )
                    if weather_data:
                        weather_list.append(weather_data)
                else:
                    self.logger.warning(f"API error for {target_date}: {response.status_code}")
                
                # Rate limiting
                import time
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                self.logger.error(f"Error fetching historical weather for {target_date}: {e}")
                continue
        
        return weather_list
    
    def _fetch_current_weather(self, location_query: str, location_name: str) -> Optional[WeatherData]:
        """Fetch current weather data"""
        try:
            params = {
                "key": self.api_key,
                "q": location_query,
                "aqi": "no"
            }
            
            response = requests.get(
                self.current_endpoint,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_current_weather_response(data, location_name)
            else:
                self.logger.warning(f"Current weather API error: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching current weather: {e}")
            return None
    
    def _fetch_forecast_weather(self, location_query: str, dates: List[date], 
                              location_name: str) -> List[WeatherData]:
        """Fetch forecast weather data (up to 14 days)"""
        weather_list = []
        
        try:
            # Calculate days ahead (WeatherAPI supports up to 14 days)
            today = date.today()
            max_forecast_date = today + timedelta(days=14)
            
            # Filter dates within forecast range
            valid_dates = [d for d in dates if d <= max_forecast_date]
            
            if not valid_dates:
                return []
            
            # Get forecast for maximum days needed
            max_days = max((d - today).days for d in valid_dates) + 1
            
            params = {
                "key": self.api_key,
                "q": location_query,
                "days": min(max_days, 14),  # API limit is 14 days
                "aqi": "no",
                "alerts": "no"
            }
            
            response = requests.get(
                self.forecast_endpoint,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                forecast_days = data.get("forecast", {}).get("forecastday", [])
                
                for forecast_day in forecast_days:
                    forecast_date_str = forecast_day.get("date")
                    forecast_date = datetime.strptime(forecast_date_str, "%Y-%m-%d").date()
                    
                    if forecast_date in valid_dates:
                        weather_data = self._parse_forecast_day(
                            forecast_day, forecast_date, location_name
                        )
                        if weather_data:
                            weather_list.append(weather_data)
            else:
                self.logger.warning(f"Forecast API error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error fetching forecast weather: {e}")
        
        return weather_list
    
    def _parse_weather_response(self, data: Dict, target_date: date, 
                              location_name: str) -> Optional[WeatherData]:
        """Parse historical weather response from WeatherAPI.com"""
        try:
            location = data.get("location", {})
            forecast_day = data.get("forecast", {}).get("forecastday", [{}])[0]
            day_data = forecast_day.get("day", {})
            astro_data = forecast_day.get("astro", {})
            
            # Extract coordinates
            latitude = location.get("lat")
            longitude = location.get("lon")
            
            # Extract weather data
            temp_max = day_data.get("maxtemp_c")
            temp_min = day_data.get("mintemp_c")
            temp_avg = day_data.get("avgtemp_c")
            precipitation = day_data.get("totalprecip_mm", 0)
            humidity = day_data.get("avghumidity")
            wind_speed = day_data.get("maxwind_kph")
            uv_index = day_data.get("uv")
            visibility = day_data.get("avgvis_km")
            
            condition = day_data.get("condition", {})
            weather_condition = condition.get("text")
            weather_icon = condition.get("icon")
            
            # Create weather data object
            weather_data = self._create_weather_data_object(
                target_date, latitude, longitude, location_name,
                temp_avg, temp_min, temp_max, precipitation, humidity,
                wind_speed, None, weather_condition, weather_icon,
                uv_index, visibility, None, None,
                astro_data.get("sunrise"), astro_data.get("sunset"),
                astro_data.get("moon_phase")
            )
            
            return weather_data
            
        except Exception as e:
            self.logger.error(f"Error parsing weather response: {e}")
            return None
    
    def _parse_current_weather_response(self, data: Dict, location_name: str) -> Optional[WeatherData]:
        """Parse current weather response from WeatherAPI.com"""
        try:
            location = data.get("location", {})
            current = data.get("current", {})
            
            # Extract coordinates
            latitude = location.get("lat")
            longitude = location.get("lon")
            
            # Extract current weather data
            temp_current = current.get("temp_c")
            feels_like = current.get("feelslike_c")
            humidity = current.get("humidity")
            wind_speed = current.get("wind_kph")
            pressure = current.get("pressure_mb")
            precipitation = current.get("precip_mm", 0)
            uv_index = current.get("uv")
            visibility = current.get("vis_km")
            cloud_cover = current.get("cloud")
            
            condition = current.get("condition", {})
            weather_condition = condition.get("text")
            weather_icon = condition.get("icon")
            
            # For current weather, use current temp as avg/min/max
            weather_data = self._create_weather_data_object(
                date.today(), latitude, longitude, location_name,
                temp_current, temp_current, temp_current, precipitation,
                humidity, wind_speed, pressure, weather_condition, weather_icon,
                uv_index, visibility, cloud_cover, feels_like
            )
            
            return weather_data
            
        except Exception as e:
            self.logger.error(f"Error parsing current weather response: {e}")
            return None
    
    def _parse_forecast_day(self, forecast_day: Dict, forecast_date: date, 
                          location_name: str) -> Optional[WeatherData]:
        """Parse forecast day data from WeatherAPI.com"""
        try:
            day_data = forecast_day.get("day", {})
            astro_data = forecast_day.get("astro", {})
            
            # Extract weather data
            temp_max = day_data.get("maxtemp_c")
            temp_min = day_data.get("mintemp_c")
            temp_avg = day_data.get("avgtemp_c")
            precipitation = day_data.get("totalprecip_mm", 0)
            humidity = day_data.get("avghumidity")
            wind_speed = day_data.get("maxwind_kph")
            uv_index = day_data.get("uv")
            visibility = day_data.get("avgvis_km")
            
            condition = day_data.get("condition", {})
            weather_condition = condition.get("text")
            weather_icon = condition.get("icon")
            
            # Note: We don't have lat/lon in forecast response, so we'll use None
            weather_data = self._create_weather_data_object(
                forecast_date, None, None, location_name,
                temp_avg, temp_min, temp_max, precipitation, humidity,
                wind_speed, None, weather_condition, weather_icon,
                uv_index, visibility, None, None,
                astro_data.get("sunrise"), astro_data.get("sunset"),
                astro_data.get("moon_phase")
            )
            
            return weather_data
            
        except Exception as e:
            self.logger.error(f"Error parsing forecast day: {e}")
            return None
    
    def _create_weather_data_object(self, target_date: date, latitude: Optional[float], 
                                  longitude: Optional[float], location_name: str,
                                  temp_avg: Optional[float], temp_min: Optional[float], 
                                  temp_max: Optional[float], precipitation: Optional[float],
                                  humidity: Optional[float], wind_speed: Optional[float], 
                                  pressure: Optional[float], weather_condition: Optional[str],
                                  weather_icon: Optional[str], uv_index: Optional[float],
                                  visibility: Optional[float], cloud_cover: Optional[int],
                                  feels_like: Optional[float] = None, 
                                  sunrise: Optional[str] = None, sunset: Optional[str] = None,
                                  moon_phase: Optional[str] = None) -> WeatherData:
        """Create WeatherData object with analysis"""
        # Calculate weather analysis
        outdoor_suitability = None
        indoor_recommendation = None
        weather_penalty = None
        
        if temp_avg is not None and precipitation is not None:
            outdoor_suitability = self.analyzer.calculate_outdoor_suitability(
                temp_avg, precipitation, wind_speed or 0
            )
            indoor_recommendation = self.analyzer.should_recommend_indoor(
                temp_avg, precipitation, wind_speed or 0
            )
            weather_penalty = self.analyzer.calculate_weather_penalty(
                temp_avg, precipitation, wind_speed or 0
            )
        
        # Create weather data object
        weather_data = WeatherData(
            date=target_date,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            temperature_avg=temp_avg,
            temperature_min=temp_min,
            temperature_max=temp_max,
            precipitation=precipitation,
            humidity=humidity,
            wind_speed=wind_speed,
            pressure=pressure,
            weather_condition=weather_condition,
            weather_icon=weather_icon,
            uv_index=uv_index,
            visibility=visibility,
            cloud_cover=cloud_cover,
            feels_like=feels_like,
            sunrise=sunrise,
            sunset=sunset,
            moon_phase=moon_phase,
            outdoor_suitability=outdoor_suitability,
            indoor_recommendation=indoor_recommendation,
            weather_penalty=weather_penalty
        )
        
        # Generate weather note
        weather_data.weather_note = self.analyzer.generate_weather_note(weather_data)
        
        return weather_data
    
    def get_weather_summary(self, weather_data_list: List[WeatherData]) -> Dict:
        """Get summary of weather data for multiple days"""
        if not weather_data_list:
            return {}
        
        # Calculate summary statistics
        temps = [w.temperature_avg for w in weather_data_list if w.temperature_avg is not None]
        precips = [w.precipitation for w in weather_data_list if w.precipitation is not None]
        outdoor_scores = [w.outdoor_suitability for w in weather_data_list if w.outdoor_suitability is not None]
        
        summary = {
            "total_days": len(weather_data_list),
            "avg_temperature": sum(temps) / len(temps) if temps else None,
            "min_temperature": min(temps) if temps else None,
            "max_temperature": max(temps) if temps else None,
            "total_precipitation": sum(precips) if precips else None,
            "avg_outdoor_suitability": sum(outdoor_scores) / len(outdoor_scores) if outdoor_scores else None,
            "rainy_days": len([p for p in precips if p > 1]) if precips else 0,
            "indoor_recommended_days": len([w for w in weather_data_list if w.indoor_recommendation]),
            "weather_conditions": [w.weather_condition for w in weather_data_list if w.weather_condition]
        }
        
        return summary