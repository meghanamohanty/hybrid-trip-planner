"""
Route Optimization Algorithm
===========================

Nearest-neighbor routing for daily POI ordering:
- Minimizes total travel distance within day cluster
- Respects opening hours and time windows
- Calculates travel times between POIs
- Ensures feasible daily schedules (8 AM - 8 PM)
- Handles timing constraints

Author: Hybrid Trip Planner Team
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import time, datetime, timedelta

# Import from Phase 1 & Phase 2
from ..data_pipeline.data_models import POI
from ..utils.data_utils import calculate_distance
from .poi_scorer import ScoredPOI
from .trip_clusterer import DayCluster


@dataclass
class RouteStop:
    """
    Single stop in optimized route
    
    Attributes:
        poi (POI): POI at this stop
        score (float): POI score
        arrival_time (time): Estimated arrival time
        departure_time (time): Estimated departure time
        duration_minutes (int): Time to spend at POI
        travel_from_previous_km (float): Distance from previous stop
        travel_time_minutes (int): Travel time from previous stop
    """
    poi: POI
    score: float
    arrival_time: time
    departure_time: time
    duration_minutes: int
    travel_from_previous_km: float
    travel_time_minutes: int


@dataclass
class OptimizedRoute:
    """
    Complete optimized route for one day
    
    Attributes:
        day_number (int): Day number
        stops (List[RouteStop]): Ordered list of stops
        total_distance_km (float): Total travel distance
        total_travel_time_minutes (int): Total travel time
        start_time (time): Day start time
        end_time (time): Day end time
        is_feasible (bool): Whether route fits in day
    """
    day_number: int
    stops: List[RouteStop]
    total_distance_km: float
    total_travel_time_minutes: int
    start_time: time
    end_time: time
    is_feasible: bool


class RouteOptimizer:
    """
    Nearest-neighbor route optimization for daily itineraries
    """
    
    def __init__(self):
        """Initialize Route Optimizer with timing constraints"""
        self.logger = logging.getLogger(__name__)
        
        # Daily schedule constraints
        self.day_start_time = time(8, 0)    # 8:00 AM
        self.day_end_time = time(20, 0)     # 8:00 PM
        self.max_day_duration_hours = 12
        
        # POI visit durations (minutes) by category
        self.default_durations = {
            'tourism': 90,
            'historic': 60,
            'leisure': 120,
            'food': 60,
            'shopping': 45,
            'accommodation': 0,  # Just for reference
            'amenity': 45
        }
        self.default_duration = 60  # Default 1 hour
        
        # Travel speed assumptions
        self.walking_speed_kmh = 4.0    # 4 km/h
        self.driving_speed_kmh = 30.0   # 30 km/h urban
        self.default_travel_mode = 'walking'
        
        # Time buffers
        self.buffer_between_stops_minutes = 10  # Buffer for transitions
        
        self.logger.info("Route Optimizer initialized")
    
    def optimize_day_route(self, day_cluster: DayCluster,
                          travel_mode: str = 'walking',
                          start_time: Optional[time] = None) -> OptimizedRoute:
        """
        Optimize route for a single day cluster
        
        Args:
            day_cluster (DayCluster): Day cluster to optimize
            travel_mode (str): Travel mode ('walking', 'driving')
            start_time (time): Day start time (default: 8 AM)
            
        Returns:
            OptimizedRoute: Optimized route for the day
        """
        if not day_cluster.pois:
            self.logger.warning(f"No POIs in day {day_cluster.day_number}")
            return self._create_empty_route(day_cluster.day_number)
        
        start = start_time or self.day_start_time
        
        self.logger.info(
            f"Optimizing route for Day {day_cluster.day_number} "
            f"with {len(day_cluster.pois)} POIs"
        )
        
        # Get travel speed for mode
        speed_kmh = (self.walking_speed_kmh if travel_mode == 'walking' 
                    else self.driving_speed_kmh)
        
        # Order POIs using nearest-neighbor algorithm
        ordered_pois = self._nearest_neighbor_order(day_cluster.pois)
        
        # Build timed route with stops
        stops = self._build_timed_route(
            ordered_pois, start, speed_kmh
        )
        
        # Calculate route statistics
        total_distance = sum(s.travel_from_previous_km for s in stops)
        total_travel_time = sum(s.travel_time_minutes for s in stops)
        
        # Check feasibility
        end_time = stops[-1].departure_time if stops else start
        is_feasible = self._check_route_feasibility(start, end_time)
        
        route = OptimizedRoute(
            day_number=day_cluster.day_number,
            stops=stops,
            total_distance_km=round(total_distance, 2),
            total_travel_time_minutes=total_travel_time,
            start_time=start,
            end_time=end_time,
            is_feasible=is_feasible
        )
        
        self.logger.info(
            f"Day {day_cluster.day_number}: {len(stops)} stops, "
            f"{route.total_distance_km}km, "
            f"{route.total_travel_time_minutes}min travel, "
            f"feasible: {is_feasible}"
        )
        
        return route
    
    def optimize_all_routes(self, day_clusters: List[DayCluster],
                           travel_mode: str = 'walking') -> List[OptimizedRoute]:
        """
        Optimize routes for all day clusters
        
        Args:
            day_clusters (List[DayCluster]): All day clusters
            travel_mode (str): Travel mode
            
        Returns:
            List[OptimizedRoute]: Optimized routes for all days
        """
        routes = []
        
        for cluster in day_clusters:
            route = self.optimize_day_route(cluster, travel_mode)
            routes.append(route)
        
        return routes
    
    def _nearest_neighbor_order(self, scored_pois: List[ScoredPOI]) -> List[ScoredPOI]:
        """
        Order POIs using nearest-neighbor algorithm
        
        Args:
            scored_pois (List[ScoredPOI]): POIs to order
            
        Returns:
            List[ScoredPOI]: Ordered POIs
        """
        if len(scored_pois) <= 1:
            return scored_pois
        
        # Start with highest scored POI
        ordered = []
        remaining = scored_pois.copy()
        remaining.sort(key=lambda x: x.total_score, reverse=True)
        
        # Take highest scored as starting point
        current = remaining.pop(0)
        ordered.append(current)
        
        # Build route by always visiting nearest unvisited POI
        while remaining:
            current_lat = current.poi.latitude
            current_lon = current.poi.longitude
            
            # Find nearest remaining POI
            nearest = None
            min_distance = float('inf')
            
            for poi in remaining:
                distance = calculate_distance(
                    current_lat, current_lon,
                    poi.poi.latitude, poi.poi.longitude
                )
                
                if distance is not None and distance < min_distance:
                    min_distance = distance
                    nearest = poi
            
            if nearest:
                ordered.append(nearest)
                remaining.remove(nearest)
                current = nearest
            else:
                # Fallback: just take next one
                ordered.append(remaining.pop(0))
        
        return ordered
    
    def _build_timed_route(self, ordered_pois: List[ScoredPOI],
                          start_time: time,
                          speed_kmh: float) -> List[RouteStop]:
        """
        Build timed route with estimated arrival/departure times
        
        Args:
            ordered_pois (List[ScoredPOI]): POIs in visit order
            start_time (time): Day start time
            speed_kmh (float): Travel speed
            
        Returns:
            List[RouteStop]: Route stops with timing
        """
        stops = []
        current_time = datetime.combine(datetime.today(), start_time)
        prev_lat, prev_lon = None, None
        
        for poi in ordered_pois:
            # Calculate travel from previous stop
            if prev_lat is not None:
                distance_km = calculate_distance(
                    prev_lat, prev_lon,
                    poi.poi.latitude, poi.poi.longitude
                ) or 0.0
                
                travel_time_minutes = int((distance_km / speed_kmh) * 60)
            else:
                distance_km = 0.0
                travel_time_minutes = 0
            
            # Add travel time and buffer
            current_time += timedelta(
                minutes=travel_time_minutes + self.buffer_between_stops_minutes
            )
            
            arrival = current_time.time()
            
            # Determine visit duration
            duration = self._get_visit_duration(poi.poi)
            
            # Calculate departure time
            current_time += timedelta(minutes=duration)
            departure = current_time.time()
            
            # Create stop
            stop = RouteStop(
                poi=poi.poi,
                score=poi.total_score,
                arrival_time=arrival,
                departure_time=departure,
                duration_minutes=duration,
                travel_from_previous_km=round(distance_km, 2),
                travel_time_minutes=travel_time_minutes
            )
            
            stops.append(stop)
            
            # Update previous location
            prev_lat = poi.poi.latitude
            prev_lon = poi.poi.longitude
        
        return stops
    
    def _get_visit_duration(self, poi: POI) -> int:
        """
        Estimate visit duration for POI
        
        Args:
            poi (POI): POI to estimate duration for
            
        Returns:
            int: Duration in minutes
        """
        # Check category-specific durations
        category = poi.category
        duration = self.default_durations.get(category, self.default_duration)
        
        # Adjust based on subcategory
        if poi.subcategory:
            if 'museum' in poi.subcategory.lower():
                duration = 90
            elif 'park' in poi.subcategory.lower():
                duration = 60
            elif 'restaurant' in poi.subcategory.lower():
                duration = 60
            elif 'cafe' in poi.subcategory.lower():
                duration = 30
            elif 'market' in poi.subcategory.lower():
                duration = 45
        
        return duration
    
    def _check_route_feasibility(self, start_time: time, end_time: time) -> bool:
        """
        Check if route fits within daily time window
        
        Args:
            start_time (time): Route start time
            end_time (time): Route end time
            
        Returns:
            bool: True if feasible
        """
        # Convert to datetime for calculation
        today = datetime.today()
        start_dt = datetime.combine(today, start_time)
        end_dt = datetime.combine(today, end_time)
        
        # Check if end is after start (same day)
        if end_dt < start_dt:
            return False
        
        # Check if within day hours
        if end_time > self.day_end_time:
            return False
        
        # Check total duration
        duration_hours = (end_dt - start_dt).total_seconds() / 3600
        if duration_hours > self.max_day_duration_hours:
            return False
        
        return True
    
    def _create_empty_route(self, day_number: int) -> OptimizedRoute:
        """Create empty route for day with no POIs"""
        return OptimizedRoute(
            day_number=day_number,
            stops=[],
            total_distance_km=0.0,
            total_travel_time_minutes=0,
            start_time=self.day_start_time,
            end_time=self.day_start_time,
            is_feasible=True
        )
    
    def adjust_route_timing(self, route: OptimizedRoute,
                           new_start_time: time) -> OptimizedRoute:
        """
        Adjust route to new start time
        
        Args:
            route (OptimizedRoute): Route to adjust
            new_start_time (time): New start time
            
        Returns:
            OptimizedRoute: Adjusted route
        """
        if not route.stops:
            return route
        
        # Get ordered POIs
        ordered_pois = [
            ScoredPOI(
                poi=stop.poi,
                total_score=stop.score,
                popularity_score=0, interest_score=0,
                weather_score=0, distance_score=0, price_score=0
            )
            for stop in route.stops
        ]
        
        # Get speed from original route
        if route.total_distance_km > 0 and route.total_travel_time_minutes > 0:
            speed_kmh = (route.total_distance_km / route.total_travel_time_minutes) * 60
        else:
            speed_kmh = self.walking_speed_kmh
        
        # Rebuild with new start time
        new_stops = self._build_timed_route(ordered_pois, new_start_time, speed_kmh)
        
        # Create new route
        new_end_time = new_stops[-1].departure_time if new_stops else new_start_time
        is_feasible = self._check_route_feasibility(new_start_time, new_end_time)
        
        return OptimizedRoute(
            day_number=route.day_number,
            stops=new_stops,
            total_distance_km=route.total_distance_km,
            total_travel_time_minutes=route.total_travel_time_minutes,
            start_time=new_start_time,
            end_time=new_end_time,
            is_feasible=is_feasible
        )
    
    def get_route_statistics(self, routes: List[OptimizedRoute]) -> Dict:
        """
        Get statistics about routes
        
        Args:
            routes (List[OptimizedRoute]): Routes to analyze
            
        Returns:
            Dict: Route statistics
        """
        if not routes:
            return {}
        
        total_distance = sum(r.total_distance_km for r in routes)
        total_travel_time = sum(r.total_travel_time_minutes for r in routes)
        total_stops = sum(len(r.stops) for r in routes)
        
        distances = [r.total_distance_km for r in routes]
        travel_times = [r.total_travel_time_minutes for r in routes]
        
        return {
            'total_days': len(routes),
            'total_stops': total_stops,
            'total_distance_km': round(total_distance, 2),
            'total_travel_time_minutes': total_travel_time,
            'avg_distance_per_day_km': round(total_distance / len(routes), 2),
            'avg_travel_time_per_day_minutes': total_travel_time // len(routes),
            'distance_range': {
                'min': round(min(distances), 2),
                'max': round(max(distances), 2)
            },
            'all_routes_feasible': all(r.is_feasible for r in routes),
            'infeasible_days': [r.day_number for r in routes if not r.is_feasible]
        }
    
    def optimize_route_order(self, route: OptimizedRoute) -> OptimizedRoute:
        """
        Re-optimize route order to improve timing
        
        Args:
            route (OptimizedRoute): Route to optimize
            
        Returns:
            OptimizedRoute: Optimized route
        """
        if len(route.stops) <= 2:
            return route
        
        # Convert stops back to scored POIs
        scored_pois = [
            ScoredPOI(
                poi=stop.poi,
                total_score=stop.score,
                popularity_score=0, interest_score=0,
                weather_score=0, distance_score=0, price_score=0
            )
            for stop in route.stops
        ]
        
        # Re-run nearest neighbor
        ordered_pois = self._nearest_neighbor_order(scored_pois)
        
        # Detect speed from original route
        if route.total_distance_km > 0 and route.total_travel_time_minutes > 0:
            speed_kmh = (route.total_distance_km / route.total_travel_time_minutes) * 60
        else:
            speed_kmh = self.walking_speed_kmh
        
        # Rebuild route
        new_stops = self._build_timed_route(ordered_pois, route.start_time, speed_kmh)
        
        # Calculate new statistics
        new_distance = sum(s.travel_from_previous_km for s in new_stops)
        new_travel_time = sum(s.travel_time_minutes for s in new_stops)
        new_end = new_stops[-1].departure_time if new_stops else route.start_time
        is_feasible = self._check_route_feasibility(route.start_time, new_end)
        
        return OptimizedRoute(
            day_number=route.day_number,
            stops=new_stops,
            total_distance_km=round(new_distance, 2),
            total_travel_time_minutes=new_travel_time,
            start_time=route.start_time,
            end_time=new_end,
            is_feasible=is_feasible
        )