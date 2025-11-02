"""
Machine Learning Engine Module
==============================

Core ML algorithms and optimization for intelligent trip planning:
- POI scoring using weighted multi-criteria evaluation
- K-means clustering for day-wise POI grouping
- Route optimization using nearest-neighbor heuristics
- Budget constraint satisfaction and cost estimation
- Weather-aware recommendation adjustments

Classes:
    POIScorer: Multi-criteria scoring algorithm for ranking Points of Interest
    TripClusterer: K-means clustering to group POIs into daily itineraries
    RouteOptimizer: Optimizes travel routes within daily clusters
    BudgetEstimator: Estimates costs and enforces budget constraints
    WeatherAdjuster: Modifies recommendations based on weather conditions

Functions:
    score_pois(): Main POI scoring function with weighted criteria
    cluster_by_days(): Groups POIs into optimal daily clusters  
    optimize_daily_route(): Creates optimal route for a day's activities
    estimate_trip_cost(): Calculates total trip cost estimation
    apply_weather_penalties(): Adjusts scores based on weather forecast
"""

# Version and module info
__version__ = "1.0.0"
__module_name__ = "ml_engine"

# Import core ML classes - Phase 2 Complete
from .poi_scorer import POIScorer, ScoredPOI
from .trip_clusterer import TripClusterer, DayCluster
from .route_optimizer import RouteOptimizer, OptimizedRoute, RouteStop
from .budget_estimator import BudgetEstimator, TripBudget, DayBudget, CostBreakdown
from .weather_adjuster import WeatherAdjuster, WeatherAdjustedPOI, WeatherAdjustedDay

# Define public API
__all__ = [
    # Core ML classes
    "POIScorer",
    "TripClusterer", 
    "RouteOptimizer",
    "BudgetEstimator",
    "WeatherAdjuster",
    
    # Data classes
    "ScoredPOI",
    "DayCluster",
    "OptimizedRoute",
    "RouteStop",
    "TripBudget",
    "DayBudget",
    "CostBreakdown",
    "WeatherAdjustedPOI",
    "WeatherAdjustedDay"
]

def get_scoring_weights():
    """
    Return default scoring weights for POI evaluation
    """
    from config import config
    return {
        "popularity": config.POPULARITY_WEIGHT,      # 0.35 - OSM tags, ratings
        "interest_match": config.INTEREST_WEIGHT,    # 0.25 - User preferences
        "weather_fit": config.WEATHER_WEIGHT,        # 0.15 - Indoor/outdoor suitability  
        "distance": config.DISTANCE_WEIGHT,          # 0.15 - Travel time penalty
        "price": config.PRICE_WEIGHT                 # 0.10 - Cost considerations
    }

def get_clustering_params():
    """
    Return default parameters for POI clustering
    """
    from config import config
    return {
        "max_radius_km": config.MAX_CLUSTER_RADIUS_KM,  # 5 km max cluster radius
        "min_pois_per_day": config.MIN_POIS_PER_DAY,    # 3 minimum POIs per day
        "max_pois_per_day": config.MAX_POIS_PER_DAY     # 8 maximum POIs per day
    }

def get_budget_distribution():
    """
    Return default budget allocation percentages
    """
    from config import config
    return {
        "hotels": config.HOTEL_BUDGET_PERCENT,          # 40% for accommodation
        "food": config.FOOD_BUDGET_PERCENT,             # 25% for meals
        "transport": config.TRANSPORT_BUDGET_PERCENT,   # 15% for transportation
        "tickets": config.TICKETS_BUDGET_PERCENT        # 20% for entry tickets
    }

def get_ml_engine_info():
    """
    Return comprehensive information about ML engine capabilities
    """
    return {
        "module": __module_name__,
        "version": __version__,
        "algorithms": {
            "scoring": "Weighted multi-criteria POI evaluation",
            "clustering": "K-means with geographical constraints", 
            "routing": "Nearest-neighbor with time windows",
            "optimization": "Constraint satisfaction with budget limits"
        },
        "features": [
            "Multi-criteria POI scoring (popularity, interest, weather, distance, price)",
            "Intelligent day-wise clustering with geographical constraints",
            "Route optimization within daily time windows", 
            "Budget-aware planning with cost estimation",
            "Weather-adaptive recommendations",
            "Real-time constraint solving"
        ],
        "scoring_weights": get_scoring_weights(),
        "clustering_params": get_clustering_params(),
        "budget_distribution": get_budget_distribution()
    }

# Module-level constants
DEFAULT_CLUSTER_METHOD = "kmeans"
DEFAULT_ROUTING_METHOD = "nearest_neighbor"
MAX_TRAVEL_TIME_MINUTES = 45
MIN_POI_SCORE = 0.3
WEATHER_PENALTY_THRESHOLD = 0.7