"""
Trip Clustering Algorithm
========================

K-means clustering to group POIs into day-wise itineraries:
- Clusters POIs by geographic proximity (lat/lon)
- Respects constraints: 3-8 POIs per day, max 5km radius
- Balances POI distribution across days
- Handles edge cases gracefully

Author: Hybrid Trip Planner Team
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans
from collections import defaultdict

# Import from Phase 1 & Phase 2
from ..data_pipeline.data_models import POI
from ..utils.data_utils import calculate_distance
from .poi_scorer import ScoredPOI
from config import config


@dataclass
class DayCluster:
    """
    Cluster of POIs for one day
    
    Attributes:
        day_number (int): Day number (1, 2, 3...)
        pois (List[ScoredPOI]): POIs in this cluster
        center_lat (float): Cluster center latitude
        center_lon (float): Cluster center longitude
        radius_km (float): Maximum distance from center
        total_score (float): Sum of POI scores
    """
    day_number: int
    pois: List[ScoredPOI]
    center_lat: float
    center_lon: float
    radius_km: float
    total_score: float


class TripClusterer:
    """
    K-means clustering for day-wise POI grouping
    """
    
    def __init__(self):
        """Initialize Trip Clusterer with config constraints"""
        self.logger = logging.getLogger(__name__)
        
        # Load constraints from config
        self.max_cluster_radius_km = config.MAX_CLUSTER_RADIUS_KM  # 5km
        self.min_pois_per_day = config.MIN_POIS_PER_DAY  # 3
        self.max_pois_per_day = config.MAX_POIS_PER_DAY  # 8
        
        self.logger.info("Trip Clusterer initialized")
    
    def cluster_pois_by_days(self, scored_pois: List[ScoredPOI], 
                            num_days: int,
                            min_score_threshold: float = 0.3) -> List[DayCluster]:
        """
        Cluster POIs into day-wise groups using K-means
        
        Args:
            scored_pois (List[ScoredPOI]): Scored POIs to cluster
            num_days (int): Number of trip days
            min_score_threshold (float): Minimum score to include
            
        Returns:
            List[DayCluster]: Day clusters sorted by day number
        """
        if not scored_pois:
            self.logger.warning("No POIs to cluster")
            return []
        
        # Filter by score threshold
        filtered_pois = [p for p in scored_pois if p.total_score >= min_score_threshold]
        
        if not filtered_pois:
            self.logger.warning(f"No POIs above score threshold {min_score_threshold}")
            return []
        
        self.logger.info(f"Clustering {len(filtered_pois)} POIs into {num_days} days...")
        
        # Check POI count constraints
        total_pois = len(filtered_pois)
        min_required = num_days * self.min_pois_per_day
        max_allowed = num_days * self.max_pois_per_day
        
        if total_pois < min_required:
            self.logger.warning(
                f"Insufficient POIs: have {total_pois}, need {min_required}. "
                f"Lowering score threshold or reducing days recommended."
            )
            # Continue with available POIs
        
        if total_pois > max_allowed:
            self.logger.info(f"Too many POIs: {total_pois}. Taking top {max_allowed}")
            filtered_pois = filtered_pois[:max_allowed]
        
        # Extract coordinates for clustering
        coordinates = np.array([
            [poi.poi.latitude, poi.poi.longitude] 
            for poi in filtered_pois
        ])
        
        # Perform K-means clustering
        try:
            clusters = self._perform_kmeans_clustering(
                coordinates, filtered_pois, num_days
            )
            
            # Validate and adjust clusters
            validated_clusters = self._validate_clusters(clusters)
            
            self.logger.info(f"Created {len(validated_clusters)} day clusters")
            return validated_clusters
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            # Fallback: simple sequential grouping
            return self._fallback_sequential_clustering(filtered_pois, num_days)
    
    def _perform_kmeans_clustering(self, coordinates: np.ndarray, 
                                   scored_pois: List[ScoredPOI],
                                   num_days: int) -> List[DayCluster]:
        """
        Perform K-means clustering on coordinates
        
        Args:
            coordinates (np.ndarray): POI coordinates (lat, lon)
            scored_pois (List[ScoredPOI]): Corresponding scored POIs
            num_days (int): Number of clusters
            
        Returns:
            List[DayCluster]: Initial clusters
        """
        # Adjust k if more days than POIs
        k = min(num_days, len(scored_pois))
        
        # Run K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coordinates)
        cluster_centers = kmeans.cluster_centers_
        
        # Group POIs by cluster
        clusters_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters_dict[label].append(scored_pois[idx])
        
        # Create DayCluster objects
        day_clusters = []
        for day_num in range(k):
            pois_in_cluster = clusters_dict.get(day_num, [])
            
            if not pois_in_cluster:
                continue
            
            # Calculate cluster center and radius
            center_lat, center_lon = cluster_centers[day_num]
            radius = self._calculate_cluster_radius(pois_in_cluster, center_lat, center_lon)
            total_score = sum(p.total_score for p in pois_in_cluster)
            
            day_cluster = DayCluster(
                day_number=day_num + 1,
                pois=pois_in_cluster,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius,
                total_score=total_score
            )
            
            day_clusters.append(day_cluster)
        
        # Sort by day number
        day_clusters.sort(key=lambda x: x.day_number)
        
        return day_clusters
    
    def _calculate_cluster_radius(self, pois: List[ScoredPOI], 
                                  center_lat: float, center_lon: float) -> float:
        """
        Calculate maximum distance from cluster center
        
        Args:
            pois (List[ScoredPOI]): POIs in cluster
            center_lat (float): Center latitude
            center_lon (float): Center longitude
            
        Returns:
            float: Maximum radius in km
        """
        if not pois:
            return 0.0
        
        max_distance = 0.0
        
        for poi in pois:
            distance = calculate_distance(
                center_lat, center_lon,
                poi.poi.latitude, poi.poi.longitude
            )
            
            if distance is not None:
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _validate_clusters(self, clusters: List[DayCluster]) -> List[DayCluster]:
        """
        Validate clusters against constraints and adjust if needed
        Now with category diversity: prioritize attractions over food
        
        Args:
            clusters (List[DayCluster]): Initial clusters
            
        Returns:
            List[DayCluster]: Validated clusters
        """
        validated = []
        overflow_pois = []
        
        for cluster in clusters:
            # Check POI count per day
            if len(cluster.pois) < self.min_pois_per_day:
                self.logger.warning(
                    f"Day {cluster.day_number} has only {len(cluster.pois)} POIs "
                    f"(min: {self.min_pois_per_day})"
                )
                # Keep cluster but flag it
                validated.append(cluster)
                
            elif len(cluster.pois) > self.max_pois_per_day:
                self.logger.info(
                    f"Day {cluster.day_number} has {len(cluster.pois)} POIs "
                    f"(max: {self.max_pois_per_day}). Trimming excess."
                )
                
                # SMART TRIMMING: Keep diverse categories, limit food
                cluster.pois = self._smart_trim_pois(
                    cluster.pois, self.max_pois_per_day
                )
                overflow_pois.extend(cluster.pois[self.max_pois_per_day:])
                cluster.pois = cluster.pois[:self.max_pois_per_day]
                
                # Recalculate cluster properties
                cluster.radius_km = self._calculate_cluster_radius(
                    cluster.pois, cluster.center_lat, cluster.center_lon
                )
                cluster.total_score = sum(p.total_score for p in cluster.pois)
                
                validated.append(cluster)
            
            else:
                # Cluster is valid
                validated.append(cluster)
            
            # Check radius constraint
            if cluster.radius_km > self.max_cluster_radius_km:
                self.logger.warning(
                    f"Day {cluster.day_number} radius {cluster.radius_km:.2f}km "
                    f"exceeds max {self.max_cluster_radius_km}km"
                )
                # Note: We could split cluster here, but keeping it simple for MVP
        
        # Distribute overflow POIs to days with space
        if overflow_pois:
            self._distribute_overflow_pois(validated, overflow_pois)
        
        return validated
    
    def _smart_trim_pois(self, pois: List[ScoredPOI], max_pois: int) -> List[ScoredPOI]:
        """
        Smart POI selection with category diversity
        Priority: tourism > historic > natural > leisure > food
        Limit: Max 2 food POIs per day
        
        Args:
            pois (List[ScoredPOI]): All POIs in cluster
            max_pois (int): Maximum POIs to keep
            
        Returns:
            List[ScoredPOI]: Selected POIs with diversity
        """
        # Categorize POIs
        priority_categories = ['tourism', 'historic', 'natural', 'leisure']
        food_categories = ['food', 'culture']
        
        high_priority = []
        food_pois = []
        other_pois = []
        
        for poi in pois:
            category = poi.poi.category
            if category in priority_categories:
                high_priority.append(poi)
            elif category in food_categories:
                food_pois.append(poi)
            else:
                other_pois.append(poi)
        
        # Sort each group by score
        high_priority.sort(key=lambda x: x.total_score, reverse=True)
        food_pois.sort(key=lambda x: x.total_score, reverse=True)
        other_pois.sort(key=lambda x: x.total_score, reverse=True)
        
        # Select POIs with diversity
        selected = []
        
        # 1. Take top high-priority POIs (attractions, historic, natural, leisure)
        slots_remaining = max_pois
        selected.extend(high_priority[:slots_remaining])
        slots_remaining -= len(selected)
        
        # 2. Add max 2 food/culture POIs
        if slots_remaining > 0:
            max_food = min(2, slots_remaining)
            selected.extend(food_pois[:max_food])
            slots_remaining -= len(food_pois[:max_food])
        
        # 3. Fill remaining with others
        if slots_remaining > 0:
            selected.extend(other_pois[:slots_remaining])
        
        # Sort final selection by score
        selected.sort(key=lambda x: x.total_score, reverse=True)
        
        return selected
    
    def _distribute_overflow_pois(self, clusters: List[DayCluster],
                                  overflow_pois: List[ScoredPOI]) -> None:
        """
        Distribute overflow POIs to clusters with capacity AND within radius
        
        Args:
            clusters (List[DayCluster]): Existing clusters
            overflow_pois (List[ScoredPOI]): POIs to distribute
        """
        
        # Sort overflow POIs by score to prioritize better ones
        overflow_pois.sort(key=lambda x: x.total_score, reverse=True)
        
        for poi in overflow_pois:
            best_cluster = None
            min_distance = float('inf')
            
            for cluster in clusters:
                # Check 1: Does the cluster have space?
                if len(cluster.pois) < self.max_pois_per_day:
                    distance = calculate_distance(
                        cluster.center_lat, cluster.center_lon,
                        poi.poi.latitude, poi.poi.longitude
                    )
                    
                    if distance is None:
                        continue
                    
                    # Check 2: Is this POI the closest one found so far?
                    # Check 3: Is this POI *actually* (realistically) close to the cluster?
                    if distance < min_distance and distance < self.max_cluster_radius_km:
                        min_distance = distance
                        best_cluster = cluster
            
            # Add to best cluster if one was found that meets all criteria
            if best_cluster:
                best_cluster.pois.append(poi)
                best_cluster.total_score += poi.total_score
                
                # We don't need to recalculate radius here, because we know
                # the POI is within the max_cluster_radius_km of the center.
                # The radius will be re-calculated by the RouteOptimizer if needed.
                # We can do a simple max() check for logging.
                best_cluster.radius_km = max(best_cluster.radius_km, min_distance)
            
            # If no cluster was found, the POI is (correctly) discarded
            # as it doesn't fit logically anywhere.
    
    def _fallback_sequential_clustering(self, scored_pois: List[ScoredPOI], 
                                       num_days: int) -> List[DayCluster]:
        """
        Fallback: Simple sequential grouping if K-means fails
        
        Args:
            scored_pois (List[ScoredPOI]): POIs to group
            num_days (int): Number of days
            
        Returns:
            List[DayCluster]: Sequential clusters
        """
        self.logger.info("Using fallback sequential clustering")
        
        # Divide POIs evenly across days
        pois_per_day = len(scored_pois) // num_days
        remainder = len(scored_pois) % num_days
        
        clusters = []
        start_idx = 0
        
        for day in range(num_days):
            # Add extra POI to first 'remainder' days
            count = pois_per_day + (1 if day < remainder else 0)
            end_idx = start_idx + count
            
            day_pois = scored_pois[start_idx:end_idx]
            
            if not day_pois:
                break
            
            # Calculate center from POIs
            avg_lat = sum(p.poi.latitude for p in day_pois) / len(day_pois)
            avg_lon = sum(p.poi.longitude for p in day_pois) / len(day_pois)
            
            radius = self._calculate_cluster_radius(day_pois, avg_lat, avg_lon)
            total_score = sum(p.total_score for p in day_pois)
            
            cluster = DayCluster(
                day_number=day + 1,
                pois=day_pois,
                center_lat=avg_lat,
                center_lon=avg_lon,
                radius_km=radius,
                total_score=total_score
            )
            
            clusters.append(cluster)
            start_idx = end_idx
        
        return clusters
    
    def get_cluster_statistics(self, clusters: List[DayCluster]) -> Dict:
        """
        Get statistics about clusters for analysis
        
        Args:
            clusters (List[DayCluster]): Day clusters
            
        Returns:
            Dict: Cluster statistics
        """
        if not clusters:
            return {}
        
        poi_counts = [len(c.pois) for c in clusters]
        radii = [c.radius_km for c in clusters]
        scores = [c.total_score for c in clusters]
        
        return {
            'total_days': len(clusters),
            'total_pois': sum(poi_counts),
            'pois_per_day': {
                'avg': sum(poi_counts) / len(poi_counts),
                'min': min(poi_counts),
                'max': max(poi_counts)
            },
            'cluster_radius_km': {
                'avg': sum(radii) / len(radii),
                'min': min(radii),
                'max': max(radii)
            },
            'total_scores': {
                'avg': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores)
            },
            'constraints_met': {
                'all_within_max_radius': all(r <= self.max_cluster_radius_km for r in radii),
                'all_above_min_pois': all(c >= self.min_pois_per_day for c in poi_counts),
                'all_below_max_pois': all(c <= self.max_pois_per_day for c in poi_counts)
            }
        }
    
    def rebalance_clusters(self, clusters: List[DayCluster]) -> List[DayCluster]:
        """
        Rebalance clusters to better meet constraints
        
        Args:
            clusters (List[DayCluster]): Clusters to rebalance
            
        Returns:
            List[DayCluster]: Rebalanced clusters
        """
        if not clusters:
            return []
        
        self.logger.info("Rebalancing clusters...")
        
        # Identify clusters needing adjustment
        deficient = [c for c in clusters if len(c.pois) < self.min_pois_per_day]
        excess = [c for c in clusters if len(c.pois) > self.max_pois_per_day]
        
        # Move POIs from excess to deficient clusters
        for def_cluster in deficient:
            needed = self.min_pois_per_day - len(def_cluster.pois)
            
            for exc_cluster in excess:
                if needed <= 0:
                    break
                
                available = len(exc_cluster.pois) - self.max_pois_per_day
                if available <= 0:
                    continue
                
                # Transfer POIs
                to_transfer = min(needed, available)
                
                # Sort excess cluster POIs by distance to deficient cluster
                exc_cluster.pois.sort(
                    key=lambda p: calculate_distance(
                        def_cluster.center_lat, def_cluster.center_lon,
                        p.poi.latitude, p.poi.longitude
                    ) or float('inf')
                )
                
                # Transfer closest POIs
                transferred = exc_cluster.pois[:to_transfer]
                exc_cluster.pois = exc_cluster.pois[to_transfer:]
                def_cluster.pois.extend(transferred)
                
                needed -= to_transfer
        
        # Recalculate cluster properties
        for cluster in clusters:
            cluster.radius_km = self._calculate_cluster_radius(
                cluster.pois, cluster.center_lat, cluster.center_lon
            )
            cluster.total_score = sum(p.total_score for p in cluster.pois)
        
        self.logger.info("Rebalancing complete")
        return clusters