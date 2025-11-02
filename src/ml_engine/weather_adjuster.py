"""
Weather-Aware Itinerary Adjustment
==================================

Adjusts POI recommendations based on weather conditions:
- Applies weather penalties to outdoor POIs
- Boosts indoor alternatives in bad weather
- Re-scores and reorders itineraries by weather
- Generates weather-appropriate recommendations
- Provides weather-aware substitutions

Author: Hybrid Trip Planner Team
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import date

# Import from Phase 1 & Phase 2
from ..data_pipeline.data_models import POI, WeatherData
from .poi_scorer import POIScorer, ScoredPOI
from .trip_clusterer import DayCluster
from .route_optimizer import OptimizedRoute


@dataclass
class WeatherAdjustedPOI:
    """
    POI with weather-adjusted score
    
    Attributes:
        scored_poi (ScoredPOI): Original scored POI
        weather_adjusted_score (float): Score after weather adjustment
        weather_penalty_applied (float): Penalty amount (0-1)
        original_score (float): Score before adjustment
        weather_recommendation (str): Weather-based recommendation
    """
    scored_poi: ScoredPOI
    weather_adjusted_score: float
    weather_penalty_applied: float
    original_score: float
    weather_recommendation: str


@dataclass
class WeatherAdjustedDay:
    """
    Day cluster with weather adjustments
    
    Attributes:
        day_number (int): Day number
        date (date): Specific date
        weather_data (WeatherData): Weather for this day
        adjusted_pois (List[WeatherAdjustedPOI]): Weather-adjusted POIs
        indoor_pois_count (int): Number of indoor POIs
        outdoor_pois_count (int): Number of outdoor POIs
        weather_suitability (float): Overall weather suitability (0-1)
        recommendations (List[str]): Weather-based recommendations
    """
    day_number: int
    date: date
    weather_data: WeatherData
    adjusted_pois: List[WeatherAdjustedPOI]
    indoor_pois_count: int
    outdoor_pois_count: int
    weather_suitability: float
    recommendations: List[str]


class WeatherAdjuster:
    """
    Weather-aware POI scoring adjustment engine
    """
    
    def __init__(self):
        """Initialize Weather Adjuster"""
        self.logger = logging.getLogger(__name__)
        
        # Weather thresholds for adjustments
        self.heavy_rain_threshold = 10.0  # mm
        self.moderate_rain_threshold = 5.0  # mm
        self.light_rain_threshold = 1.0  # mm
        
        self.cold_temp_threshold = 10.0  # Celsius
        self.hot_temp_threshold = 35.0  # Celsius
        self.comfortable_temp_min = 18.0
        self.comfortable_temp_max = 28.0
        
        # Adjustment multipliers
        self.outdoor_penalty_multiplier = 0.7  # 30% penalty max
        self.indoor_boost_multiplier = 1.2  # 20% boost max
        
        self.logger.info("Weather Adjuster initialized")
    
    def adjust_pois_for_weather(self, scored_pois: List[ScoredPOI],
                                weather_data: WeatherData,
                                day_date: date) -> List[WeatherAdjustedPOI]:
        """
        Adjust POI scores based on weather conditions
        
        Args:
            scored_pois (List[ScoredPOI]): Scored POIs to adjust
            weather_data (WeatherData): Weather data for the day
            day_date (date): Date of the day
            
        Returns:
            List[WeatherAdjustedPOI]: Weather-adjusted POIs
        """
        if not scored_pois:
            return []
        
        self.logger.info(
            f"Adjusting {len(scored_pois)} POIs for weather on {day_date}"
        )
        
        adjusted_pois = []
        
        for scored_poi in scored_pois:
            adjusted = self._adjust_single_poi(scored_poi, weather_data)
            adjusted_pois.append(adjusted)
        
        # Re-sort by adjusted scores
        adjusted_pois.sort(key=lambda x: x.weather_adjusted_score, reverse=True)
        
        return adjusted_pois
    
    def adjust_day_clusters(self, day_clusters: List[DayCluster],
                           weather_data_list: List[WeatherData],
                           start_date: date) -> List[WeatherAdjustedDay]:
        """
        Adjust all day clusters based on weather forecast
        
        Args:
            day_clusters (List[DayCluster]): Day clusters to adjust
            weather_data_list (List[WeatherData]): Weather for each day
            start_date (date): Trip start date
            
        Returns:
            List[WeatherAdjustedDay]: Weather-adjusted days
        """
        if not day_clusters:
            return []
        
        adjusted_days = []
        
        for cluster in day_clusters:
            # Find matching weather data
            day_index = cluster.day_number - 1
            if day_index < len(weather_data_list):
                weather = weather_data_list[day_index]
                
                # Calculate date for this day
                from datetime import timedelta
                day_date = start_date + timedelta(days=day_index)
                
                adjusted_day = self._adjust_single_day(
                    cluster, weather, day_date
                )
                adjusted_days.append(adjusted_day)
            else:
                self.logger.warning(
                    f"No weather data for day {cluster.day_number}"
                )
        
        return adjusted_days
    
    def _adjust_single_poi(self, scored_poi: ScoredPOI,
                          weather_data: WeatherData) -> WeatherAdjustedPOI:
        """
        Adjust single POI score based on weather
        
        Args:
            scored_poi (ScoredPOI): POI to adjust
            weather_data (WeatherData): Weather data
            
        Returns:
            WeatherAdjustedPOI: Adjusted POI
        """
        original_score = scored_poi.total_score
        poi = scored_poi.poi
        
        # Check if POI is indoor/outdoor
        is_outdoor = poi.outdoor if poi.outdoor is not None else None
        
        # Calculate weather penalty
        penalty = self._calculate_weather_penalty(weather_data, is_outdoor)
        
        # Apply adjustment
        if is_outdoor is True:
            # Outdoor POI - apply penalty
            adjusted_score = original_score * (1.0 - penalty * self.outdoor_penalty_multiplier)
        elif is_outdoor is False:
            # Indoor POI - apply boost in bad weather
            if penalty > 0.3:  # Significant bad weather
                boost = penalty * 0.5  # Partial boost
                adjusted_score = original_score * (1.0 + boost * 0.2)  # Max 20% boost
            else:
                adjusted_score = original_score
        else:
            # Unknown - apply moderate penalty
            adjusted_score = original_score * (1.0 - penalty * 0.5)
        
        # Generate recommendation
        recommendation = self._generate_poi_recommendation(
            poi, weather_data, penalty
        )
        
        return WeatherAdjustedPOI(
            scored_poi=scored_poi,
            weather_adjusted_score=max(adjusted_score, 0.0),
            weather_penalty_applied=penalty,
            original_score=original_score,
            weather_recommendation=recommendation
        )
    
    def _calculate_weather_penalty(self, weather_data: WeatherData,
                                   is_outdoor: Optional[bool]) -> float:
        """
        Calculate weather penalty score
        
        Args:
            weather_data (WeatherData): Weather data
            is_outdoor (bool): Whether POI is outdoor
            
        Returns:
            float: Penalty (0-1, higher is worse weather)
        """
        # Use pre-calculated penalty if available
        if weather_data.weather_penalty is not None:
            return weather_data.weather_penalty
        
        # Calculate from weather conditions
        penalty = 0.0
        
        temp = weather_data.temperature_avg or 20
        precip = weather_data.precipitation or 0
        
        # Temperature penalties
        if temp < self.cold_temp_threshold:
            temp_penalty = (self.cold_temp_threshold - temp) / 10.0
            penalty += min(temp_penalty, 0.3)
        elif temp > self.hot_temp_threshold:
            temp_penalty = (temp - self.hot_temp_threshold) / 10.0
            penalty += min(temp_penalty, 0.2)
        
        # Precipitation penalties
        if precip > self.heavy_rain_threshold:
            penalty += 0.5
        elif precip > self.moderate_rain_threshold:
            penalty += 0.3
        elif precip > self.light_rain_threshold:
            penalty += 0.1
        
        return min(penalty, 1.0)
    
    def _generate_poi_recommendation(self, poi: POI,
                                    weather_data: WeatherData,
                                    penalty: float) -> str:
        """
        Generate weather-based recommendation for POI
        
        Args:
            poi (POI): POI
            weather_data (WeatherData): Weather data
            penalty (float): Weather penalty
            
        Returns:
            str: Recommendation message
        """
        if penalty < 0.2:
            return "Good weather for visiting"
        
        is_outdoor = poi.outdoor
        
        if is_outdoor is True:
            if penalty > 0.5:
                return "Not recommended - poor weather for outdoor activity"
            elif penalty > 0.3:
                return "Caution - weather may affect outdoor experience"
            else:
                return "Weather acceptable but not ideal"
        elif is_outdoor is False:
            if penalty > 0.3:
                return "Excellent indoor alternative during bad weather"
            else:
                return "Good indoor option"
        else:
            if penalty > 0.4:
                return "Consider indoor alternatives"
            else:
                return "Weather conditions moderate"
    
    def _adjust_single_day(self, day_cluster: DayCluster,
                          weather_data: WeatherData,
                          day_date: date) -> WeatherAdjustedDay:
        """
        Adjust single day cluster for weather
        
        Args:
            day_cluster (DayCluster): Day cluster
            weather_data (WeatherData): Weather data
            day_date (date): Date
            
        Returns:
            WeatherAdjustedDay: Adjusted day
        """
        # Adjust all POIs in cluster
        adjusted_pois = self.adjust_pois_for_weather(
            day_cluster.pois, weather_data, day_date
        )
        
        # Count indoor/outdoor POIs
        indoor_count = sum(
            1 for ap in adjusted_pois 
            if ap.scored_poi.poi.outdoor is False
        )
        outdoor_count = sum(
            1 for ap in adjusted_pois 
            if ap.scored_poi.poi.outdoor is True
        )
        
        # Calculate overall suitability
        suitability = weather_data.outdoor_suitability or 0.5
        
        # Generate day recommendations
        recommendations = self._generate_day_recommendations(
            weather_data, indoor_count, outdoor_count, adjusted_pois
        )
        
        return WeatherAdjustedDay(
            day_number=day_cluster.day_number,
            date=day_date,
            weather_data=weather_data,
            adjusted_pois=adjusted_pois,
            indoor_pois_count=indoor_count,
            outdoor_pois_count=outdoor_count,
            weather_suitability=suitability,
            recommendations=recommendations
        )
    
    def _generate_day_recommendations(self, weather_data: WeatherData,
                                     indoor_count: int,
                                     outdoor_count: int,
                                     adjusted_pois: List[WeatherAdjustedPOI]) -> List[str]:
        """
        Generate recommendations for the day
        
        Args:
            weather_data (WeatherData): Weather data
            indoor_count (int): Number of indoor POIs
            outdoor_count (int): Number of outdoor POIs
            adjusted_pois (List[WeatherAdjustedPOI]): Adjusted POIs
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        temp = weather_data.temperature_avg or 20
        precip = weather_data.precipitation or 0
        
        # Temperature recommendations
        if temp < self.cold_temp_threshold:
            recommendations.append("Cold day - dress warmly and limit outdoor time")
        elif temp > self.hot_temp_threshold:
            recommendations.append("Hot day - stay hydrated and seek shade/AC frequently")
        
        # Precipitation recommendations
        if precip > self.heavy_rain_threshold:
            recommendations.append("Heavy rain expected - focus on indoor activities")
            if outdoor_count > indoor_count:
                recommendations.append(
                    "Consider postponing outdoor POIs to another day"
                )
        elif precip > self.moderate_rain_threshold:
            recommendations.append("Rain expected - carry umbrella and waterproof gear")
        elif precip > self.light_rain_threshold:
            recommendations.append("Light rain possible - have umbrella handy")
        
        # Balance recommendations
        if outdoor_count > 0 and precip > self.moderate_rain_threshold:
            recommendations.append(
                f"Schedule {outdoor_count} outdoor activities for breaks in weather"
            )
        
        # Indoor/outdoor balance
        if weather_data.outdoor_suitability and weather_data.outdoor_suitability > 0.7:
            recommendations.append("Great weather for outdoor exploration")
        
        return recommendations
    
    def reorder_pois_by_weather(self, adjusted_day: WeatherAdjustedDay) -> WeatherAdjustedDay:
        """
        Reorder POIs within day based on weather timing
        
        Args:
            adjusted_day (WeatherAdjustedDay): Day to reorder
            
        Returns:
            WeatherAdjustedDay: Reordered day
        """
        # Simple strategy: outdoor activities when weather is better
        # Indoor activities during worse weather (afternoon storms, etc.)
        
        pois = adjusted_day.adjusted_pois.copy()
        
        # Separate indoor and outdoor
        indoor = [p for p in pois if p.scored_poi.poi.outdoor is False]
        outdoor = [p for p in pois if p.scored_poi.poi.outdoor is True]
        unknown = [p for p in pois if p.scored_poi.poi.outdoor is None]
        
        # Reorder based on weather pattern
        weather = adjusted_day.weather_data
        
        if weather.precipitation and weather.precipitation > self.moderate_rain_threshold:
            # Rain expected - prioritize indoor
            reordered = indoor + unknown + outdoor
        else:
            # Good weather - balance indoor/outdoor
            reordered = []
            # Alternate indoor/outdoor for variety
            while indoor or outdoor or unknown:
                if outdoor:
                    reordered.append(outdoor.pop(0))
                if indoor:
                    reordered.append(indoor.pop(0))
                if unknown:
                    reordered.append(unknown.pop(0))
        
        # Update day with reordered POIs
        adjusted_day.adjusted_pois = reordered
        
        return adjusted_day
    
    def suggest_alternatives(self, poi: ScoredPOI,
                           all_pois: List[ScoredPOI],
                           weather_data: WeatherData) -> List[ScoredPOI]:
        """
        Suggest weather-appropriate alternatives to a POI
        
        Args:
            poi (ScoredPOI): POI to find alternatives for
            all_pois (List[ScoredPOI]): All available POIs
            weather_data (WeatherData): Weather data
            
        Returns:
            List[ScoredPOI]: Alternative POIs
        """
        # If outdoor POI in bad weather, suggest indoor alternatives
        if poi.poi.outdoor is True and weather_data.indoor_recommendation:
            # Find indoor POIs in same category
            alternatives = [
                p for p in all_pois
                if p.poi.outdoor is False
                and p.poi.category == poi.poi.category
                and p.poi.osm_id != poi.poi.osm_id
            ]
            
            # Sort by score
            alternatives.sort(key=lambda x: x.total_score, reverse=True)
            return alternatives[:3]  # Top 3 alternatives
        
        return []
    
    def get_weather_statistics(self, adjusted_days: List[WeatherAdjustedDay]) -> Dict:
        """
        Get weather adjustment statistics
        
        Args:
            adjusted_days (List[WeatherAdjustedDay]): Adjusted days
            
        Returns:
            Dict: Weather statistics
        """
        if not adjusted_days:
            return {}
        
        total_pois = sum(len(day.adjusted_pois) for day in adjusted_days)
        
        # Calculate average penalties
        all_penalties = [
            poi.weather_penalty_applied
            for day in adjusted_days
            for poi in day.adjusted_pois
        ]
        avg_penalty = sum(all_penalties) / len(all_penalties) if all_penalties else 0
        
        # Count adjustments
        significant_adjustments = sum(
            1 for p in all_penalties if p > 0.3
        )
        
        return {
            'total_days': len(adjusted_days),
            'total_pois': total_pois,
            'avg_weather_penalty': round(avg_penalty, 3),
            'significant_adjustments': significant_adjustments,
            'days_with_poor_weather': sum(
                1 for day in adjusted_days 
                if day.weather_suitability < 0.5
            ),
            'days_with_good_weather': sum(
                1 for day in adjusted_days 
                if day.weather_suitability >= 0.7
            ),
            'total_indoor_pois': sum(day.indoor_pois_count for day in adjusted_days),
            'total_outdoor_pois': sum(day.outdoor_pois_count for day in adjusted_days),
            'avg_suitability': sum(
                day.weather_suitability for day in adjusted_days
            ) / len(adjusted_days)
        }