"""
OpenStreetMap Data Loader
========================

Fetches Points of Interest (POIs) from OpenStreetMap using the Overpass API.
Fixed to use POI from data_models to avoid circular imports.

Author: Hybrid Trip Planner Team
"""

import logging
import time
import requests
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote
import json
import os

# Import POI from data_models (not from this module)
from .data_models import POI

# Import configuration and utilities (only what we need)
from config import config
from ..utils import CacheManager, ErrorHandler


class OSMQueryBuilder:
    """
    Helper class for building Overpass API queries
    Constructs structured queries for different POI types and geographic areas
    """
    
    
    @staticmethod
    def build_poi_query(bbox: Tuple[float, float, float, float], 
                       poi_types: Optional[List[str]] = None) -> str:
        """
        Build query to fetch POIs within bounding box
        
        Args:
            bbox (Tuple): Bounding box (south, west, north, east)
            poi_types (List[str]): List of POI types to fetch
            
        Returns:
            str: Overpass API query string
        """
        if poi_types is None:
            poi_types = ["tourism", "amenity", "leisure", "historic", "shop", "natural"]
        
        south, west, north, east = bbox
        
        # Build query components for different POI categories
        query_parts = []
        
        # Tourism attractions (HIGH PRIORITY)
        if "tourism" in poi_types:
            query_parts.append(f'node["tourism"]({south},{west},{north},{east});')
            query_parts.append(f'way["tourism"]({south},{west},{north},{east});')
        
        # Natural features (beaches, peaks, etc.) - HIGH PRIORITY
        if "natural" in poi_types:
            natural_types = ["beach", "peak", "park", "cave_entrance", "waterfall", 
                           "hot_spring", "spring", "cliff", "bay"]
            for natural in natural_types:
                query_parts.append(f'node["natural"="{natural}"]({south},{west},{north},{east});')
                query_parts.append(f'way["natural"="{natural}"]({south},{west},{north},{east});')
        
        # Religious sites (temples, churches, mosques) - HIGH PRIORITY
        if "amenity" in poi_types:
            # Temples and places of worship
            query_parts.append(f'node["amenity"="place_of_worship"]({south},{west},{north},{east});')
            query_parts.append(f'way["amenity"="place_of_worship"]({south},{west},{north},{east});')
            
            # Culture & entertainment (theatre, cinema, etc.)
            culture_types = ["theatre", "cinema", "arts_centre", "library"]
            for culture in culture_types:
                query_parts.append(f'node["amenity"="{culture}"]({south},{west},{north},{east});')
                query_parts.append(f'way["amenity"="{culture}"]({south},{west},{north},{east});')
            
            # Food (REDUCED - only famous restaurants/cafes, limit query results)
            food_types = ["restaurant", "cafe"]
            for food in food_types:
                query_parts.append(f'node["amenity"="{food}"]({south},{west},{north},{east});')
                query_parts.append(f'way["amenity"="{food}"]({south},{west},{north},{east});')
        
        # Leisure facilities
        if "leisure" in poi_types:
            leisure_types = ["park", "garden", "playground", "sports_centre", 
                           "swimming_pool", "beach_resort", "marina", "stadium"]
            for leisure in leisure_types:
                query_parts.append(f'node["leisure"="{leisure}"]({south},{west},{north},{east});')
                query_parts.append(f'way["leisure"="{leisure}"]({south},{west},{north},{east});')
        
        # Historic sites (HIGH PRIORITY)
        if "historic" in poi_types:
            query_parts.append(f'node["historic"]({south},{west},{north},{east});')
            query_parts.append(f'way["historic"]({south},{west},{north},{east});')
        
        # Shopping (REDUCED - only tourist-relevant)
        if "shop" in poi_types:
            shop_types = ["mall", "marketplace", "gift", "souvenir"]
            for shop in shop_types:
                query_parts.append(f'node["shop"="{shop}"]({south},{west},{north},{east});')
                query_parts.append(f'way["shop"="{shop}"]({south},{west},{north},{east});')
        
        # Combine all query parts
        query = f"""
        [out:json][timeout:30];
        (
          {chr(10).join(query_parts)}
        );
        out center meta;
        """
        
        return query.strip()


class OSMDataLoader:
    """
    Main class for loading POI data from OpenStreetMap
    Handles API communication, data processing, and caching
    """
    
    def __init__(self):
        """Initialize OSM Data Loader with configuration and caching"""
        self.logger = logging.getLogger(__name__)
        
        # Overpass API configuration
        self.base_url = "http://overpass-api.de/api/interpreter"
        self.backup_urls = [
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.openstreetmap.ru/api/interpreter"
        ]
        
        # Rate limiting configuration
        self.request_delay = config.OSM_RATE_LIMIT_DELAY
        self.timeout = config.OSM_API_TIMEOUT
        self.max_retries = 3
        
        self.logger.info("OSM Data Loader initialized")
    
    def get_city_bounds(self, city_name: str, country: str = "India") -> Optional[Tuple[float, float, float, float]]:
        """
        Get bounding box coordinates for a city using Nominatim API
        
        Args:
            city_name (str): Name of the city
            country (str): Country name
            
        Returns:
            Tuple[float, float, float, float]: Bounding box (south, west, north, east)
            None: If city not found or error occurred
        """
        cache_manager = CacheManager()
        error_handler = ErrorHandler()
        
        # Check cache first
        cache_key = f"city_bounds_{city_name}_{country}"
        cached_bounds = cache_manager.get(cache_key)
        if cached_bounds:
            self.logger.debug(f"Retrieved city bounds from cache for {city_name}")
            return cached_bounds
        
        try:
            # Use Nominatim API for geocoding
            url = f"https://nominatim.openstreetmap.org/search?q={quote(city_name)},{quote(country)}&format=json&limit=1"
            headers = {"User-Agent": "Hybrid-Trip-Planner/1.0"}
            
            self.logger.info(f"Fetching city bounds for {city_name} from Nominatim...")
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            results = response.json()
            
            if not results:
                self.logger.warning(f"No city boundaries found for {city_name} using Nominatim")
                return None
            
            # Extract bounding box: [minlat, maxlat, minlon, maxlon]
            bbox = results[0].get("boundingbox")
            if not bbox:
                self.logger.warning(f"Bounding box not found in Nominatim response for {city_name}")
                return None

            # Convert to (south, west, north, east) -> (minlat, minlon, maxlat, maxlon)
            bounds = (float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3]))
            
            # Cache the result
            cache_manager.set(cache_key, bounds, ttl=86400)  # Cache for 24 hours
            self.logger.info(f"Retrieved bounds for {city_name}: {bounds}")
            
            return bounds
            
        except requests.exceptions.RequestException as e:
            error_handler.handle_error(f"Error getting city bounds for {city_name} from Nominatim", e)
            return None
        except (IndexError, KeyError) as e:
            error_handler.handle_error(f"Error parsing Nominatim response for {city_name}", e)
            return None
    
    def fetch_pois(self, city_name: str, poi_types: Optional[List[str]] = None,
                   max_results: int = 500) -> List[POI]:
        """
        Fetch Points of Interest for a given city
        
        Args:
            city_name (str): Name of the city
            poi_types (List[str]): Types of POIs to fetch
            max_results (int): Maximum number of results to return
            
        Returns:
            List[POI]: List of POI objects
        """
        try:
            # Get city bounding box
            bounds = self.get_city_bounds(city_name)
            if not bounds:
                self.logger.error(f"Could not get bounds for city: {city_name}")
                return []
            
            # Build POI query
            query = OSMQueryBuilder.build_poi_query(bounds, poi_types)
            
            # Execute query
            response = self._execute_query(query)
            if not response:
                return []
            
            # Parse POIs from response
            pois = self._parse_pois(response)
            
            # Filter and validate POIs
            valid_pois = self._filter_and_validate_pois(pois, max_results)
            
            self.logger.info(f"Fetched {len(valid_pois)} POIs for {city_name}")
            return valid_pois
            
        except Exception as e:
            self.logger.error(f"Error fetching POIs for {city_name}: {e}")
            return []
    
    def _execute_query(self, query: str) -> Optional[Dict]:
        """Execute Overpass API query with retry logic and error handling"""
        urls_to_try = [self.base_url] + self.backup_urls
        
        for attempt in range(self.max_retries):
            for url in urls_to_try:
                try:
                    self.logger.debug(f"Executing query (attempt {attempt + 1}) on {url}")
                    
                    # Add rate limiting delay
                    time.sleep(self.request_delay)
                    
                    # Execute request
                    response = requests.post(
                        url,
                        data={"data": query},
                        timeout=self.timeout,
                        headers={"User-Agent": "Hybrid-Trip-Planner/1.0"}
                    )
                    
                    # Check response status
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 429:  # Rate limited
                        self.logger.warning("Rate limited, waiting longer...")
                        time.sleep(5)
                        continue
                    else:
                        self.logger.warning(f"API returned status {response.status_code}")
                        continue
                        
                except requests.exceptions.Timeout:
                    self.logger.warning(f"Timeout on {url}")
                    continue
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Request error on {url}: {e}")
                    continue
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON decode error: {e}")
                    continue
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        self.logger.error("All query attempts failed")
        return None
    
    
    def _parse_pois(self, response: Dict) -> List[POI]:
        """Parse POI objects from API response"""
        pois = []
        elements = response.get("elements", [])
        
        for element in elements:
            try:
                poi = self._parse_single_poi(element)
                if poi:
                    pois.append(poi)
            except Exception as e:
                self.logger.debug(f"Error parsing POI element: {e}")
                continue
        
        return pois
    
    def _parse_single_poi(self, element: Dict) -> Optional[POI]:
        """Parse a single POI from an OSM element"""
        tags = element.get("tags", {})
        
        # Skip if no name
        name = tags.get("name")
        if not name:
            return None
        
        # Get coordinates
        if element.get("type") == "node":
            lat, lon = element.get("lat"), element.get("lon")
        elif element.get("center"):
            lat, lon = element["center"]["lat"], element["center"]["lon"]
        else:
            return None
        
        # Determine category and subcategory
        category, subcategory = self._determine_category(tags)
        if not category:
            return None
        
        # Create POI object
        poi = POI(
            osm_id=str(element.get("id", "")),
            name=name,
            category=category,
            subcategory=subcategory,
            latitude=lat,
            longitude=lon,
            address=self._extract_address(tags),
            phone=tags.get("phone"),
            website=tags.get("website"),
            opening_hours=tags.get("opening_hours"),
            rating=self._extract_rating(tags),
            price_level=self._extract_price_level(tags),
            tags=tags,
            outdoor=self._is_outdoor_activity(tags, category),
            wheelchair_accessible=self._is_wheelchair_accessible(tags),
            fee_required=self._requires_fee(tags)
        )
        
        return poi
    
    def _determine_category(self, tags: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Determine POI category and subcategory from OSM tags - TOURIST FOCUSED"""
        
        # Non-tourist amenities to EXCLUDE
        NON_TOURIST_AMENITIES = [
            "hospital", "clinic", "doctors", "dentist", "pharmacy", "veterinary",
            "school", "college", "university", "kindergarten",
            "bank", "atm", "post_office", "police", "fire_station",
            "courthouse", "prison", "fuel", "car_wash", "parking",
            "toilets", "waste_disposal", "recycling", "social_facility"
        ]
        
        # Priority 1: Tourism (highest priority for tourists)
        if "tourism" in tags:
            tourism_type = tags["tourism"]
            # Exclude accommodation (not visit spots)
            if tourism_type not in ["hotel", "hostel", "guesthouse", "motel", "apartment"]:
                return "tourism", tourism_type
        
        # Priority 2: Historic sites (always tourist-relevant)
        if "historic" in tags:
            historic_type = tags["historic"]
            return "historic", historic_type
        
        # Priority 3: Leisure (tourist-friendly only)
        if "leisure" in tags:
            leisure_type = tags["leisure"]
            tourist_leisure = [
                "park", "garden", "playground", "sports_centre",
                "swimming_pool", "water_park", "beach_resort", "marina",
                "nature_reserve", "amusement_arcade", "bowling_alley",
                "miniature_golf", "golf_course", "stadium"
            ]
            if leisure_type in tourist_leisure:
                return "leisure", leisure_type
        
        # Priority 4: Amenity (HEAVILY FILTERED - only food, culture, entertainment)
        if "amenity" in tags:
            amenity_type = tags["amenity"]
            
            # EXCLUDE non-tourist amenities
            if amenity_type in NON_TOURIST_AMENITIES:
                return None, None
            
            # Food & drink (tourist-relevant)
            if amenity_type in ["restaurant", "cafe", "bar", "pub", "fast_food", "food_court", "ice_cream"]:
                return "food", amenity_type
            
            # Culture & entertainment (tourist-relevant)
            if amenity_type in ["theatre", "cinema", "arts_centre", "community_centre", "library", "place_of_worship"]:
                return "culture", amenity_type
            
            # Entertainment (tourist-relevant)
            if amenity_type in ["nightclub", "casino", "spa", "sauna"]:
                return "entertainment", amenity_type
            
            # If not in tourist categories, exclude it
            return None, None
        
        # Priority 5: Shopping (tourist shops only)
        if "shop" in tags:
            shop_type = tags["shop"]
            tourist_shops = [
                "mall", "department_store", "supermarket", "marketplace",
                "gift", "souvenir", "art", "books", "clothes", "shoes",
                "jewelry", "toys", "sports", "outdoor", "craft"
            ]
            if shop_type in tourist_shops:
                return "shopping", shop_type
        
        # Priority 6: Natural features
        if "natural" in tags:
            natural_type = tags["natural"]
            tourist_natural = [
                "beach", "peak", "park", "cave_entrance", "waterfall", "spring",
                "hot_spring", "geyser", "cliff", "bay"
            ]
            if natural_type in tourist_natural:
                return "natural", natural_type
        
        return None, None
    
    def _extract_address(self, tags: Dict) -> Optional[str]:
        """Extract address from OSM tags"""
        address_parts = []
        
        # Add house number and street
        if "addr:housenumber" in tags:
            address_parts.append(tags["addr:housenumber"])
        if "addr:street" in tags:
            address_parts.append(tags["addr:street"])
        
        # Add city and postcode
        if "addr:city" in tags:
            address_parts.append(tags["addr:city"])
        if "addr:postcode" in tags:
            address_parts.append(tags["addr:postcode"])
        
        return ", ".join(address_parts) if address_parts else None
    
    def _extract_rating(self, tags: Dict) -> Optional[float]:
        """Extract rating from OSM tags"""
        if "rating" in tags:
            try:
                return float(tags["rating"])
            except ValueError:
                pass
        return None
    
    def _extract_price_level(self, tags: Dict) -> Optional[int]:
        """Extract price level from OSM tags"""
        if tags.get("fee") == "yes":
            return 3  # Assume expensive if fee required
        elif tags.get("fee") == "no":
            return 1  # Assume cheap if no fee
        if "charge" in tags or "fee" in tags:
            return 2  # Medium price if charges mentioned
        return None
    
    def _is_outdoor_activity(self, tags: Dict, category: str) -> Optional[bool]:
        """
        Determine if POI is primarily outdoor. Assumes tourist attractions
        are OUTDOOR unless explicitly tagged as INDOOR.
        """
        
        # 1. High-confidence INDOOR tags
        if tags.get("indoor") == "yes":
            return False
        
        # 2. High-confidence INDOOR categories
        indoor_categories = ['food', 'shopping'] # Culture = museums, theatres
        if category in indoor_categories:
            return False

        # 3. High-confidence INDOOR subcategories
        indoor_subcategories = [
            "museum", "gallery", "theatre", "cinema", "library",
            "restaurant", "cafe", "bar", "pub", "mall", "arts_centre"
        ]
        # Check 'culture' subcategories here instead
        if category == 'culture' and tags.get('amenity') in indoor_subcategories:
             return False
        # Treat temples as "unknown" (None) so they don't get a wrong classification
        if category == 'culture' and tags.get('amenity') == 'place_of_worship':
             return None # <-- This is better than False

        # 4. High-confidence OUTDOOR tags
        if tags.get("indoor") == "no":
            return True

        # 5. High-confidence OUTDOOR categories (beaches, parks, etc.)
        outdoor_categories = ['natural', 'leisure']
        if category in outdoor_categories:
            return True
            
        outdoor_subcategories = [
            "beach", "park", "garden", "viewpoint", "attraction",
            "picnic_site", "monument", "memorial", "ruins", "archaeological_site"
        ]
        if tags.get("natural") == "beach":
            return True
        if tags.get("tourism") in outdoor_subcategories:
            return True
        if tags.get("historic") in outdoor_subcategories:
            return True

        # 6. Default for remaining 'tourism' and 'historic':
        # This will correctly classify most viewpoints, attractions, etc. as OUTDOOR.
        if category in ['tourism', 'historic']:
            return True

        # Fallback for anything else (e.g., 'amenity' POIs not in our lists)
        return None
    
    def _is_wheelchair_accessible(self, tags: Dict) -> Optional[bool]:
        """Check wheelchair accessibility"""
        wheelchair = tags.get("wheelchair")
        if wheelchair == "yes":
            return True
        elif wheelchair == "no":
            return False
        return None
    
    def _requires_fee(self, tags: Dict) -> Optional[bool]:
        """Check if entry fee is required"""
        fee = tags.get("fee")
        if fee == "yes":
            return True
        elif fee == "no":
            return False
        return None
    
    def _filter_and_validate_pois(self, pois: List[POI], max_results: int) -> List[POI]:
        """Filter and validate POIs"""
        valid_pois = []
        
        for poi in pois:
            # Basic validation
            if not poi.name or not poi.latitude or not poi.longitude:
                continue
            
            # Coordinate validation
            if not (-90 <= poi.latitude <= 90) or not (-180 <= poi.longitude <= 180):
                continue
            
            # Category validation
            if not poi.category or not poi.subcategory:
                continue
            
            valid_pois.append(poi)
            
            # Limit results
            if len(valid_pois) >= max_results:
                break
        
        self.logger.info(f"Filtered {len(pois)} POIs down to {len(valid_pois)} valid POIs")
        return valid_pois
    
    def get_poi_categories(self) -> Dict[str, List[str]]:
        """Get available POI categories and subcategories - TOURIST FOCUSED"""
        return {
            "tourism": ["attraction", "museum", "gallery", "monument", "viewpoint", 
                       "zoo", "theme_park", "aquarium", "artwork"],
            "food": ["restaurant", "cafe", "bar", "pub", "fast_food", "food_court", "ice_cream"],
            "leisure": ["park", "garden", "playground", "sports_centre", 
                       "swimming_pool", "beach_resort", "marina", "stadium", "water_park"],
            "historic": ["monument", "memorial", "archaeological_site", "castle", 
                        "palace", "fort", "ruins"],
            "shopping": ["mall", "supermarket", "marketplace", "gift", "souvenir", "craft"],
            "culture": ["theatre", "cinema", "arts_centre", "library"],
            "entertainment": ["nightclub", "casino", "spa", "sauna"],
            "natural": ["beach", "peak", "waterfall", "cave_entrance", "hot_spring"]
        }