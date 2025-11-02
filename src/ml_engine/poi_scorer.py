"""
POI Scoring Algorithm
====================

Multi-criteria POI scoring using weighted evaluation:
- Popularity (35%): OSM prominence, ratings, category importance
- Interest Match (25%): Alignment with user preferences
- Weather Fit (15%): Outdoor suitability based on weather
- Distance (15%): Proximity to trip center/other POIs
- Price (10%): Budget alignment

Author: Hybrid Trip Planner Team
"""

import logging
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import from Phase 1
from ..data_pipeline.data_models import POI, WeatherData
from ..utils.data_utils import calculate_distance
from config import config


@dataclass
class ScoredPOI:
    """
    POI with calculated score and breakdown
    
    Attributes:
        poi (POI): Original POI object
        total_score (float): Final weighted score (0-1)
        popularity_score (float): Popularity component
        interest_score (float): Interest match component
        weather_score (float): Weather fit component
        distance_score (float): Distance component
        price_score (float): Price component
    """
    poi: POI
    total_score: float
    popularity_score: float
    interest_score: float
    weather_score: float
    distance_score: float
    price_score: float


class POIScorer:
    """
    Main POI scoring engine using multi-criteria evaluation
    """
    
    def __init__(self):
        """Initialize POI Scorer with configuration weights"""
        self.logger = logging.getLogger(__name__)
        
        # Load weights from config
        self.weights = {
            'popularity': config.POPULARITY_WEIGHT,      # 0.35
            'interest': config.INTEREST_WEIGHT,          # 0.25
            'weather': config.WEATHER_WEIGHT,            # 0.15
            'distance': config.DISTANCE_WEIGHT,          # 0.15
            'price': config.PRICE_WEIGHT                 # 0.10
        }
        
        # Category importance scores (higher = more popular category)
        self.category_importance = {
            'tourism': 1.0,
            'historic': 0.95,
            'natural': 0.9,       # <-- ADD THIS (For Beaches, Viewpoints)
            'culture': 0.9,
            'leisure': 0.85,
            'food': 0.80,
            'shopping': 0.70,
            'accommodation': 0.60,
            'amenity': 0.75
        }
        
        self.logger.info("POI Scorer initialized with config weights")
    
    def score_pois(self, pois: List[POI], 
                   center_lat: float, center_lon: float,
                   user_interests: Optional[List[str]] = None,
                   weather_data: Optional[WeatherData] = None,
                   budget_per_poi: Optional[float] = None) -> List[ScoredPOI]:
        """
        Score a list of POIs using multi-criteria evaluation
        
        Args:
            pois (List[POI]): POIs to score
            center_lat (float): Trip center latitude
            center_lon (float): Trip center longitude
            user_interests (List[str]): User interest categories
            weather_data (WeatherData): Weather data for scoring
            budget_per_poi (float): Average budget per POI
            
        Returns:
            List[ScoredPOI]: Scored POIs sorted by total score (descending)
        """
        if not pois:
            self.logger.warning("No POIs to score")
            return []
        
        self.logger.info(f"Scoring {len(pois)} POIs...")
        
        scored_pois = []
        
        for poi in pois:
            try:
                # Calculate individual scores
                pop_score = self._calculate_popularity_score(poi)
                int_score = self._calculate_interest_score(poi, user_interests)
                weather_score = self._calculate_weather_score(poi, weather_data)
                dist_score = self._calculate_distance_score(poi, center_lat, center_lon)
                price_score = self._calculate_price_score(poi, budget_per_poi)
                
                # Calculate weighted total score
                total_score = (
                    pop_score * self.weights['popularity'] +
                    int_score * self.weights['interest'] +
                    weather_score * self.weights['weather'] +
                    dist_score * self.weights['distance'] +
                    price_score * self.weights['price']
                )
                
                # Create scored POI
                scored_poi = ScoredPOI(
                    poi=poi,
                    total_score=total_score,
                    popularity_score=pop_score,
                    interest_score=int_score,
                    weather_score=weather_score,
                    distance_score=dist_score,
                    price_score=price_score
                )
                
                scored_pois.append(scored_poi)
                
            except Exception as e:
                self.logger.warning(f"Error scoring POI {poi.name}: {e}")
                continue
        
        # Sort by total score (descending)
        scored_pois.sort(key=lambda x: x.total_score, reverse=True)
        
        self.logger.info(f"Scored {len(scored_pois)} POIs successfully")
        return scored_pois
    
    def _calculate_popularity_score(self, poi: POI) -> float:
        """
        Calculate popularity score based on OSM tags and ratings
        
        Args:
            poi (POI): POI object
            
        Returns:
            float: Popularity score (0-1)
        """
        score = 0.0
        
        # Category importance (40% of popularity)
        category_score = self.category_importance.get(poi.category, 0.5)
        score += category_score * 0.4
        
        # Rating if available (30% of popularity)
        if poi.rating is not None:
            rating_normalized = min(poi.rating / 5.0, 1.0)  # Normalize to 0-1
            score += rating_normalized * 0.3
        else:
            score += 0.5 * 0.3  # Default middle score
        
        # OSM tag richness (20% of popularity)
        if poi.tags:
            # More tags = more information = likely more important
            tag_richness = min(len(poi.tags) / 20.0, 1.0)  # Cap at 20 tags
            score += tag_richness * 0.2
        else:
            score += 0.3 * 0.2
        
        # Has website/phone (10% of popularity)
        contact_score = 0.0
        if poi.website:
            contact_score += 0.5
        if poi.phone:
            contact_score += 0.5
        score += contact_score * 0.1
        
        return min(score, 1.0)
    
    def _calculate_interest_score(self, poi: POI, 
                                  user_interests: Optional[List[str]]) -> float:
        """
        Calculate interest match score
        
        Args:
            poi (POI): POI object
            user_interests (List[str]): User interest categories
            
        Returns:
            float: Interest score (0-1)
        """
        if not user_interests:
            # No preferences = neutral score
            return 0.5
        
        # Normalize user interests to lowercase
        interests_lower = [i.lower() for i in user_interests]
        
        # Check category match
        if poi.category.lower() in interests_lower:
            return 1.0
        
        # Check subcategory match
        if poi.subcategory and poi.subcategory.lower() in interests_lower:
            return 0.9
        
        # Check tag matches
        if poi.tags:
            tag_matches = 0
            for interest in interests_lower:
                for tag_key, tag_value in poi.tags.items():
                    if interest in tag_key.lower() or interest in str(tag_value).lower():
                        tag_matches += 1
                        break
            
            if tag_matches > 0:
                return min(0.5 + (tag_matches * 0.2), 0.8)
        
        # No match = low score
        return 0.5
    
    def _calculate_weather_score(self, poi: POI, 
                                 weather_data: Optional[WeatherData]) -> float:
        """
        Calculate weather fit score (penalty for outdoor in bad weather)
        
        Args:
            poi (POI): POI object
            weather_data (WeatherData): Weather data
            
        Returns:
            float: Weather score (0-1, higher is better)
        """
        # No weather data = neutral score
        if not weather_data:
            return 0.5
        
        # Indoor POIs not affected by weather
        if poi.outdoor is False:
            return 1.0
        
        # Use weather penalty if available
        if weather_data.weather_penalty is not None:
            # Outdoor POIs get penalized
            if poi.outdoor is True:
                return 1.0 - weather_data.weather_penalty
            else:
                # Unknown indoor/outdoor = moderate penalty
                return 1.0 - (weather_data.weather_penalty * 0.5)
        
        # Fallback: calculate from temperature and precipitation
        temp = weather_data.temperature_avg or 20
        precip = weather_data.precipitation or 0
        
        penalty = 0.0
        
        # Temperature penalties
        if temp < 10:
            penalty += 0.3
        elif temp > 35:
            penalty += 0.2
        
        # Precipitation penalties
        if precip > 10:
            penalty += 0.4
        elif precip > 5:
            penalty += 0.2
        
        if poi.outdoor is True:
            return max(1.0 - penalty, 0.0)
        else:
            # Unknown = moderate penalty
            return max(1.0 - (penalty * 0.5), 0.3)
    
    def _calculate_distance_score(self, poi: POI, 
                                  center_lat: float, center_lon: float) -> float:
        """
        Calculate distance score (penalty for far POIs)
        
        Args:
            poi (POI): POI object
            center_lat (float): Center latitude
            center_lon (float): Center longitude
            
        Returns:
            float: Distance score (0-1, higher is closer)
        """
        # Calculate distance
        distance_km = calculate_distance(
            center_lat, center_lon,
            poi.latitude, poi.longitude
        )
        
        if distance_km is None:
            return 0.5  # Default if calculation fails
        
        # Score based on distance (exponential decay)
        # 0 km = 1.0, 5 km = 0.5, 10 km = 0.25, 20 km = 0.1
        max_distance = 20.0  # km
        
        if distance_km <= 1.0:
            return 1.0
        elif distance_km >= max_distance:
            return 0.1
        else:
            # Exponential decay
            score = math.exp(-distance_km / 5.0)
            return max(score, 0.1)
    
    def _calculate_price_score(self, poi: POI, 
                              budget_per_poi: Optional[float]) -> float:
        """
        Calculate price score (penalty if over budget)
        
        Args:
            poi (POI): POI object
            budget_per_poi (float): Average budget per POI
            
        Returns:
            float: Price score (0-1, higher is better value)
        """
        # No budget constraint = neutral score
        if budget_per_poi is None:
            return 0.5
        
        # Free POIs get bonus
        if poi.fee_required is False:
            return 1.0
        
        # Estimate cost based on price level
        if poi.price_level is not None:
            # price_level: 1=cheap, 2=moderate, 3=expensive, 4=very expensive
            estimated_cost = poi.price_level * (budget_per_poi / 2)
            
            if estimated_cost <= budget_per_poi:
                # Within budget
                return 1.0 - (estimated_cost / budget_per_poi) * 0.3
            else:
                # Over budget - penalty
                over_factor = estimated_cost / budget_per_poi
                return max(0.3 / over_factor, 0.1)
        
        # Fee required but no price info
        if poi.fee_required is True:
            return 0.6  # Moderate penalty
        
        # Unknown - assume moderate cost
        return 0.5
    
    def get_top_pois(self, scored_pois: List[ScoredPOI], 
                    n: int = 10, 
                    min_score: float = 0.3) -> List[ScoredPOI]:
        """
        Get top N POIs with minimum score threshold
        
        Args:
            scored_pois (List[ScoredPOI]): Scored POIs
            n (int): Number of top POIs to return
            min_score (float): Minimum score threshold
            
        Returns:
            List[ScoredPOI]: Top N POIs
        """
        # Filter by minimum score
        filtered = [poi for poi in scored_pois if poi.total_score >= min_score]
        
        # Return top N
        return filtered[:n]
    
    def get_pois_by_category(self, scored_pois: List[ScoredPOI], 
                            category: str) -> List[ScoredPOI]:
        """
        Get scored POIs filtered by category
        
        Args:
            scored_pois (List[ScoredPOI]): Scored POIs
            category (str): Category to filter by
            
        Returns:
            List[ScoredPOI]: Filtered POIs
        """
        return [poi for poi in scored_pois if poi.poi.category == category]
    
    def get_score_statistics(self, scored_pois: List[ScoredPOI]) -> Dict:
        """
        Get scoring statistics for analysis
        
        Args:
            scored_pois (List[ScoredPOI]): Scored POIs
            
        Returns:
            Dict: Score statistics
        """
        if not scored_pois:
            return {}
        
        scores = [poi.total_score for poi in scored_pois]
        
        return {
            'total_pois': len(scored_pois),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'scores_above_0.5': len([s for s in scores if s >= 0.5]),
            'scores_above_0.7': len([s for s in scores if s >= 0.7]),
            'top_category': self._get_top_category(scored_pois),
            'weight_contributions': {
                'popularity_avg': sum(p.popularity_score for p in scored_pois) / len(scored_pois),
                'interest_avg': sum(p.interest_score for p in scored_pois) / len(scored_pois),
                'weather_avg': sum(p.weather_score for p in scored_pois) / len(scored_pois),
                'distance_avg': sum(p.distance_score for p in scored_pois) / len(scored_pois),
                'price_avg': sum(p.price_score for p in scored_pois) / len(scored_pois)
            }
        }
    
    def _get_top_category(self, scored_pois: List[ScoredPOI]) -> Optional[str]:
        """Get category with highest average score"""
        if not scored_pois:
            return None
        
        category_scores = {}
        category_counts = {}
        
        for poi in scored_pois:
            cat = poi.poi.category
            if cat not in category_scores:
                category_scores[cat] = 0.0
                category_counts[cat] = 0
            
            category_scores[cat] += poi.total_score
            category_counts[cat] += 1
        
        # Calculate averages
        category_avgs = {
            cat: category_scores[cat] / category_counts[cat]
            for cat in category_scores
        }
        
        return max(category_avgs, key=category_avgs.get)