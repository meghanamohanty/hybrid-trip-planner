"""
Data Validation and API Testing Module
=====================================

Validates data from OSM and Weather APIs.
Fixed to import from data_models to avoid circular imports.

Author: Hybrid Trip Planner Team
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import requests
import time

# Import data classes from data_models (not from loaders)
from .data_models import POI, WeatherData
from .weather_loader import WeatherDataLoader

# Import utilities directly (no circular import)
from config import config


class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Data class representing a validation issue"""
    level: ValidationLevel
    category: str
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Data class representing validation results"""
    is_valid: bool
    score: float
    total_items: int
    valid_items: int
    issues: List[ValidationIssue]
    summary: Dict
    recommendations: List[str]


class APITester:
    """Helper class for testing API connections and health"""
    
    def __init__(self):
        """Initialize API Tester"""
        self.logger = logging.getLogger(__name__)
    
    def test_osm_api(self) -> Tuple[bool, Dict]:
        """Test OpenStreetMap Overpass API connection"""
        test_results = {
            "api_name": "OpenStreetMap Overpass API",
            "endpoint": "http://overpass-api.de/api/interpreter",
            "test_timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "response_time_ms": None,
            "error_message": None
        }
        
        try:
            # Simple test query for a small area
            test_query = """
            [out:json][timeout:10];
            node["name"="Test"][place=city](17.6,83.2,17.7,83.3);
            out 1;
            """
            
            start_time = time.time()
            response = requests.post(
                "http://overpass-api.de/api/interpreter",
                data={"data": test_query},
                timeout=15,
                headers={"User-Agent": "Hybrid-Trip-Planner-Test/1.0"}
            )
            end_time = time.time()
            
            test_results["response_time_ms"] = int((end_time - start_time) * 1000)
            
            if response.status_code == 200:
                # Try to parse JSON
                data = response.json()
                if "elements" in data:
                    test_results["status"] = "success"
                    self.logger.info("OSM API test passed")
                    return True, test_results
                else:
                    test_results["status"] = "error"
                    test_results["error_message"] = "Invalid JSON structure"
            else:
                test_results["status"] = "error"
                test_results["error_message"] = f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            test_results["status"] = "timeout"
            test_results["error_message"] = "Request timeout"
        except requests.exceptions.RequestException as e:
            test_results["status"] = "connection_error"
            test_results["error_message"] = str(e)
        except Exception as e:
            test_results["status"] = "unknown_error"
            test_results["error_message"] = str(e)
        
        self.logger.warning(f"OSM API test failed: {test_results['error_message']}")
        return False, test_results
    
    def test_weather_api(self) -> Tuple[bool, Dict]:
        """Test WeatherAPI.com connection"""
        test_results = {
            "api_name": "WeatherAPI.com",
            "endpoint": "http://api.weatherapi.com/v1/current.json",
            "test_timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "response_time_ms": None,
            "error_message": None,
            "api_key_status": "unknown"
        }
        
        try:
            # Test with Visakhapatnam coordinates
            test_params = {
                "key": config.WEATHER_API_KEY,
                "q": "17.6868,83.2185",  # Visakhapatnam
                "aqi": "no"
            }
            
            start_time = time.time()
            response = requests.get(
                "http://api.weatherapi.com/v1/current.json",
                params=test_params,
                timeout=15
            )
            end_time = time.time()
            
            test_results["response_time_ms"] = int((end_time - start_time) * 1000)
            
            if response.status_code == 200:
                # Try to parse JSON and validate structure
                data = response.json()
                if "current" in data and "location" in data:
                    test_results["status"] = "success"
                    test_results["api_key_status"] = "valid"
                    self.logger.info("Weather API test passed")
                    return True, test_results
                else:
                    test_results["status"] = "error"
                    test_results["error_message"] = "Invalid JSON structure"
            elif response.status_code == 401:
                test_results["status"] = "auth_error"
                test_results["api_key_status"] = "invalid"
                test_results["error_message"] = "Invalid API key"
            elif response.status_code == 403:
                test_results["status"] = "quota_exceeded"
                test_results["error_message"] = "API quota exceeded"
            else:
                test_results["status"] = "error"
                test_results["error_message"] = f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            test_results["status"] = "timeout"
            test_results["error_message"] = "Request timeout"
        except requests.exceptions.RequestException as e:
            test_results["status"] = "connection_error"
            test_results["error_message"] = str(e)
        except Exception as e:
            test_results["status"] = "unknown_error"
            test_results["error_message"] = str(e)
        
        self.logger.warning(f"Weather API test failed: {test_results['error_message']}")
        return False, test_results
    
    def run_all_api_tests(self) -> Dict[str, Tuple[bool, Dict]]:
        """Run all API connection tests"""
        results = {}
        
        self.logger.info("Starting API connection tests...")
        
        # Test OSM API
        osm_success, osm_results = self.test_osm_api()
        results["osm"] = (osm_success, osm_results)
        
        # Test Weather API
        weather_success, weather_results = self.test_weather_api()
        results["weather"] = (weather_success, weather_results)
        
        # Overall status
        all_success = osm_success and weather_success
        self.logger.info(f"API tests completed. Overall status: {'PASS' if all_success else 'FAIL'}")
        
        return results


class DataQualityChecker:
    """Analyzes data quality and completeness"""
    
    def __init__(self):
        """Initialize Data Quality Checker"""
        self.logger = logging.getLogger(__name__)
    
    def calculate_poi_quality_score(self, pois: List[POI]) -> float:
        """Calculate quality score for POI data"""
        if not pois:
            return 0.0
        
        total_score = 0.0
        
        for poi in pois:
            poi_score = 0.0
            
            # Required fields (40 points)
            if poi.name and poi.latitude and poi.longitude and poi.category:
                poi_score += 40
            
            # Coordinate validity (20 points)
            if self._validate_coordinates(poi.latitude, poi.longitude):
                poi_score += 20
            
            # Optional but important fields (20 points)
            optional_fields = [poi.address, poi.phone, poi.website, poi.opening_hours]
            filled_optional = sum(1 for field in optional_fields if field)
            poi_score += (filled_optional / len(optional_fields)) * 20
            
            # Data richness (20 points)
            richness_fields = [poi.rating, poi.price_level, poi.outdoor, 
                             poi.wheelchair_accessible, poi.fee_required]
            filled_richness = sum(1 for field in richness_fields if field is not None)
            poi_score += (filled_richness / len(richness_fields)) * 20
            
            total_score += poi_score
        
        return total_score / len(pois)
    
    def calculate_weather_quality_score(self, weather_data: List[WeatherData]) -> float:
        """Calculate quality score for weather data"""
        if not weather_data:
            return 0.0
        
        total_score = 0.0
        
        for weather in weather_data:
            weather_score = 0.0
            
            # Required fields (50 points)
            if (weather.date and weather.temperature_avg is not None and 
                weather.precipitation is not None):
                weather_score += 50
            
            # Temperature data completeness (20 points)
            temp_fields = [weather.temperature_min, weather.temperature_max, weather.feels_like]
            temp_score = sum(1 for field in temp_fields if field is not None)
            weather_score += (temp_score / len(temp_fields)) * 20
            
            # Additional weather data (20 points)
            additional_fields = [weather.humidity, weather.wind_speed, weather.pressure, 
                               weather.uv_index, weather.visibility]
            additional_score = sum(1 for field in additional_fields if field is not None)
            weather_score += (additional_score / len(additional_fields)) * 20
            
            # Analysis completeness (10 points)
            analysis_fields = [weather.outdoor_suitability, weather.indoor_recommendation, 
                             weather.weather_penalty]
            analysis_score = sum(1 for field in analysis_fields if field is not None)
            weather_score += (analysis_score / len(analysis_fields)) * 10
            
            total_score += weather_score
        
        return total_score / len(weather_data)
    
    def _validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Validate coordinates are within valid ranges"""
        try:
            lat = float(latitude)
            lon = float(longitude)
            return -90 <= lat <= 90 and -180 <= lon <= 180
        except (ValueError, TypeError):
            return False


class DataValidator:
    """Main class for validating POI and weather data"""
    
    def __init__(self):
        """Initialize Data Validator"""
        self.logger = logging.getLogger(__name__)
        self.api_tester = APITester()
        self.quality_checker = DataQualityChecker()
    
    def validate_poi_data(self, pois: List[POI]) -> ValidationResult:
        """Validate POI data from OSM loader"""
        issues = []
        valid_count = 0
        
        self.logger.info(f"Validating {len(pois)} POI records...")
        
        for i, poi in enumerate(pois):
            poi_issues = self._validate_single_poi(poi, i)
            issues.extend(poi_issues)
            
            # Count as valid if no critical or error issues
            has_critical_issues = any(issue.level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR] 
                                    for issue in poi_issues)
            if not has_critical_issues:
                valid_count += 1
        
        # Calculate quality score
        quality_score = self.quality_checker.calculate_poi_quality_score(pois)
        
        # Generate summary
        summary = self._generate_poi_summary(pois, issues)
        
        # Generate recommendations
        recommendations = self._generate_poi_recommendations(issues, quality_score)
        
        # Determine overall validity
        critical_issues = [issue for issue in issues if issue.level == ValidationLevel.CRITICAL]
        is_valid = len(critical_issues) == 0 and valid_count > 0
        
        result = ValidationResult(
            is_valid=is_valid,
            score=quality_score,
            total_items=len(pois),
            valid_items=valid_count,
            issues=issues,
            summary=summary,
            recommendations=recommendations
        )
        
        self.logger.info(f"POI validation completed. Score: {quality_score:.1f}/100, Valid: {valid_count}/{len(pois)}")
        return result
    
    def validate_weather_data(self, weather_data: List[WeatherData]) -> ValidationResult:
        """Validate weather data from Weather API"""
        issues = []
        valid_count = 0
        
        self.logger.info(f"Validating {len(weather_data)} weather records...")
        
        for i, weather in enumerate(weather_data):
            weather_issues = self._validate_single_weather(weather, i)
            issues.extend(weather_issues)
            
            # Count as valid if no critical or error issues
            has_critical_issues = any(issue.level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR] 
                                    for issue in weather_issues)
            if not has_critical_issues:
                valid_count += 1
        
        # Calculate quality score
        quality_score = self.quality_checker.calculate_weather_quality_score(weather_data)
        
        # Generate summary
        summary = self._generate_weather_summary(weather_data, issues)
        
        # Generate recommendations
        recommendations = self._generate_weather_recommendations(issues, quality_score)
        
        # Determine overall validity
        critical_issues = [issue for issue in issues if issue.level == ValidationLevel.CRITICAL]
        is_valid = len(critical_issues) == 0 and valid_count > 0
        
        result = ValidationResult(
            is_valid=is_valid,
            score=quality_score,
            total_items=len(weather_data),
            valid_items=valid_count,
            issues=issues,
            summary=summary,
            recommendations=recommendations
        )
        
        self.logger.info(f"Weather validation completed. Score: {quality_score:.1f}/100, Valid: {valid_count}/{len(weather_data)}")
        return result
    
    def _validate_single_poi(self, poi: POI, index: int) -> List[ValidationIssue]:
        """Validate a single POI object"""
        issues = []
        
        # Required field validation
        if not poi.name or len(poi.name.strip()) == 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="required_field",
                message=f"POI #{index}: Missing or empty name",
                field="name",
                value=poi.name,
                suggestion="Ensure POI has a valid name"
            ))
        
        # Coordinate validation
        if poi.latitude is None or poi.longitude is None:
            issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="coordinates",
                message=f"POI #{index}: Missing coordinates",
                field="latitude/longitude",
                value=f"{poi.latitude}, {poi.longitude}",
                suggestion="Ensure POI has valid latitude and longitude"
            ))
        elif not self.quality_checker._validate_coordinates(poi.latitude, poi.longitude):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="coordinates",
                message=f"POI #{index}: Invalid coordinates",
                field="latitude/longitude",
                value=f"{poi.latitude}, {poi.longitude}",
                suggestion="Check coordinate values are within valid ranges"
            ))
        
        return issues
    
    def _validate_single_weather(self, weather: WeatherData, index: int) -> List[ValidationIssue]:
        """Validate a single weather data object"""
        issues = []
        
        # Required field validation
        if not weather.date:
            issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="required_field",
                message=f"Weather #{index}: Missing date",
                field="date",
                suggestion="Ensure weather data has valid date"
            ))
        
        if weather.temperature_avg is None:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="required_field",
                message=f"Weather #{index}: Missing temperature data",
                field="temperature_avg",
                suggestion="Temperature data is required for weather analysis"
            ))
        elif not -50 <= weather.temperature_avg <= 60:  # Reasonable temperature range
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="data_range",
                message=f"Weather #{index}: Temperature outside normal range",
                field="temperature_avg",
                value=weather.temperature_avg,
                suggestion="Verify temperature value is correct"
            ))
        
        return issues
    
    def _generate_poi_summary(self, pois: List[POI], issues: List[ValidationIssue]) -> Dict:
        """Generate summary statistics for POI validation"""
        categories = {}
        has_address = 0
        
        for poi in pois:
            # Count categories
            if poi.category:
                categories[poi.category] = categories.get(poi.category, 0) + 1
            
            # Count optional fields
            if poi.address:
                has_address += 1
        
        # Count issues by level
        issue_counts = {level.value: 0 for level in ValidationLevel}
        for issue in issues:
            issue_counts[issue.level.value] += 1
        
        return {
            "total_pois": len(pois),
            "categories": categories,
            "completeness": {
                "with_address": has_address
            },
            "issues_by_level": issue_counts
        }
    
    def _generate_weather_summary(self, weather_data: List[WeatherData], issues: List[ValidationIssue]) -> Dict:
        """Generate summary statistics for weather validation"""
        if not weather_data:
            return {}
        
        # Temperature statistics
        temps = [w.temperature_avg for w in weather_data if w.temperature_avg is not None]
        
        # Count issues by level
        issue_counts = {level.value: 0 for level in ValidationLevel}
        for issue in issues:
            issue_counts[issue.level.value] += 1
        
        return {
            "total_weather_records": len(weather_data),
            "temperature_stats": {
                "avg": sum(temps) / len(temps) if temps else None,
                "min": min(temps) if temps else None,
                "max": max(temps) if temps else None
            },
            "issues_by_level": issue_counts
        }
    
    def _generate_poi_recommendations(self, issues: List[ValidationIssue], quality_score: float) -> List[str]:
        """Generate recommendations for POI data improvement"""
        recommendations = []
        
        critical_count = len([i for i in issues if i.level == ValidationLevel.CRITICAL])
        error_count = len([i for i in issues if i.level == ValidationLevel.ERROR])
        
        if critical_count > 0:
            recommendations.append(f"Fix {critical_count} critical issues before proceeding")
        
        if error_count > 0:
            recommendations.append(f"Address {error_count} data errors for better quality")
        
        if quality_score < 70:
            recommendations.append("Consider enhancing data with additional fields")
        
        return recommendations
    
    def _generate_weather_recommendations(self, issues: List[ValidationIssue], quality_score: float) -> List[str]:
        """Generate recommendations for weather data improvement"""
        recommendations = []
        
        critical_count = len([i for i in issues if i.level == ValidationLevel.CRITICAL])
        
        if critical_count > 0:
            recommendations.append(f"Fix {critical_count} critical weather data issues")
        
        if quality_score < 80:
            recommendations.append("Consider using additional weather data sources")
        
        return recommendations
    
    def run_full_validation(self, city_name: str = "Visakhapatnam") -> Dict:
        """Run complete validation including API tests and data validation"""
        self.logger.info(f"Running full validation for {city_name}...")
        
        validation_report = {
            "test_timestamp": datetime.now().isoformat(),
            "city_tested": city_name,
            "api_tests": {},
            "data_validation": {},
            "overall_status": "unknown",
            "recommendations": []
        }
        
        try:
            # 1. Test API connections
            self.logger.info("Testing API connections...")
            api_results = self.api_tester.run_all_api_tests()
            validation_report["api_tests"] = api_results
            
            # 2. Test data fetching and validation
            if api_results["osm"][0] and api_results["weather"][0]:
                self.logger.info("APIs working, testing data fetching...")
                
                # Test POI data
                from .osm_loader import OSMDataLoader
                osm_loader = OSMDataLoader()
                test_pois = osm_loader.fetch_pois(city_name, max_results=10)
                
                if test_pois:
                    poi_validation = self.validate_poi_data(test_pois)
                    validation_report["data_validation"]["poi"] = {
                        "validation_result": poi_validation,
                        "sample_size": len(test_pois)
                    }
                
                # Test weather data
                weather_loader = WeatherDataLoader()
                test_weather = weather_loader.get_weather_for_trip(
                    17.6868, 83.2185,  # Visakhapatnam coordinates
                    date.today(),
                    3  # 3 days
                )
                
                if test_weather:
                    weather_validation = self.validate_weather_data(test_weather)
                    validation_report["data_validation"]["weather"] = {
                        "validation_result": weather_validation,
                        "sample_size": len(test_weather)
                    }
            
            # 3. Determine overall status
            all_apis_working = all(result[0] for result in api_results.values())
            data_validations_passed = True
            
            if "poi" in validation_report["data_validation"]:
                data_validations_passed &= validation_report["data_validation"]["poi"]["validation_result"].is_valid
            
            if "weather" in validation_report["data_validation"]:
                data_validations_passed &= validation_report["data_validation"]["weather"]["validation_result"].is_valid
            
            if all_apis_working and data_validations_passed:
                validation_report["overall_status"] = "success"
                validation_report["recommendations"].append("Phase 1 data pipeline is ready for Phase 2 ML development")
            elif all_apis_working:
                validation_report["overall_status"] = "partial"
                validation_report["recommendations"].append("APIs working but data quality needs improvement")
            else:
                validation_report["overall_status"] = "failed"
                validation_report["recommendations"].append("Fix API connectivity issues before proceeding")
            
        except Exception as e:
            validation_report["overall_status"] = "error"
            validation_report["error"] = str(e)
        
        self.logger.info(f"Full validation completed. Status: {validation_report['overall_status']}")
        return validation_report