#!/usr/bin/env python3
"""
Phase 1 Testing Script
=====================

Comprehensive test to verify all Phase 1 components work correctly.
Tests imports, API connections, data fetching, and validation.

Usage:
    python test_phase1.py

Author: Hybrid Trip Planner Team
"""

import sys
import os
import traceback
from datetime import date, timedelta

# Test configuration
TEST_CITY = "Mumbai"
TEST_COORDINATES = (19.0760, 72.8777)  # Mumbai coordinates


def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")


def print_test_result(test_name: str, success: bool, message: str = ""):
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"    {message}")


def test_imports():
    """Test if all modules can be imported correctly"""
    print_test_header("Module Imports")
    
    import_tests = [
        ("config", "from config import config"),
        ("utils.cache_manager", "from src.utils.cache_manager import CacheManager"),
        ("utils.error_handler", "from src.utils.error_handler import ErrorHandler"),
        ("utils.data_utils", "from src.utils.data_utils import validate_coordinates, calculate_distance"),
        ("utils.performance_monitor", "from src.utils.performance_monitor import measure_time, PerformanceMonitor"),
        ("data_pipeline.osm_loader", "from src.data_pipeline.osm_loader import OSMDataLoader, POI"),
        ("data_pipeline.weather_loader", "from src.data_pipeline.weather_loader import WeatherDataLoader, WeatherData"),
        ("data_pipeline.data_validator", "from src.data_pipeline.data_validator import DataValidator, ValidationLevel")
    ]
    
    results = {}
    
    for module_name, import_statement in import_tests:
        try:
            exec(import_statement)
            print_test_result(f"Import {module_name}", True)
            results[module_name] = True
        except Exception as e:
            print_test_result(f"Import {module_name}", False, str(e))
            results[module_name] = False
    
    return results


def test_configuration():
    """Test configuration loading"""
    print_test_header("Configuration")
    
    try:
        from config import config
        
        # Test basic config access
        app_name = config.APP_NAME
        gemini_key = config.GEMINI_API_KEY
        
        print_test_result("Config loading", True, f"App: {app_name}")
        print_test_result("Gemini API key", bool(gemini_key), f"Key length: {len(gemini_key) if gemini_key else 0}")
        
        return True
        
    except Exception as e:
        print_test_result("Configuration", False, str(e))
        return False


def test_utilities():
    """Test utility functions"""
    print_test_header("Utility Functions")
    
    try:
        from src.utils.data_utils import validate_coordinates, calculate_distance
        from src.utils.cache_manager import CacheManager
        from src.utils.error_handler import ErrorHandler
        
        # Test coordinate validation
        valid_coords = validate_coordinates(17.6868, 83.2185)
        invalid_coords = validate_coordinates(91.0, 181.0)
        
        print_test_result("Coordinate validation (valid)", valid_coords)
        print_test_result("Coordinate validation (invalid)", not invalid_coords)
        
        # Test distance calculation
        distance = calculate_distance(17.6868, 83.2185, 17.7231, 83.3210)
        print_test_result("Distance calculation", distance is not None, f"Distance: {distance} km")
        
        # Test cache manager
        cache = CacheManager(max_size=10, default_ttl=60)
        cache.set("test_key", "test_value")
        cached_value = cache.get("test_key")
        print_test_result("Cache manager", cached_value == "test_value")
        
        # Test error handler
        error_handler = ErrorHandler()
        print_test_result("Error handler", True, "Initialized successfully")
        
        return True
        
    except Exception as e:
        print_test_result("Utilities", False, str(e))
        traceback.print_exc()
        return False


def test_osm_api():
    """Test OpenStreetMap API connection and data fetching"""
    print_test_header("OpenStreetMap API")
    
    try:
        from src.data_pipeline.osm_loader import OSMDataLoader
        
        # Create OSM loader
        osm_loader = OSMDataLoader()
        print_test_result("OSM Loader creation", True)
        
        # Test city bounds fetching
        print("Fetching city bounds...")
        bounds = osm_loader.get_city_bounds(TEST_CITY)
        print_test_result("City bounds fetching", bounds is not None, f"Bounds: {bounds}")
        
        if bounds:
            # Test POI fetching (limited to 5 for quick test)
            print("Fetching sample POIs...")
            pois = osm_loader.fetch_pois(TEST_CITY, max_results=5)
            print_test_result("POI fetching", len(pois) > 0, f"Found {len(pois)} POIs")
            
            if pois:
                # Show sample POI
                sample_poi = pois[0]
                print(f"    Sample POI: {sample_poi.name} ({sample_poi.category})")
                return True
        
        return False
        
    except Exception as e:
        print_test_result("OSM API", False, str(e))
        traceback.print_exc()
        return False


def test_weather_api():
    """Test WeatherAPI.com connection and data fetching"""
    print_test_header("WeatherAPI.com")
    
    try:
        from src.data_pipeline.weather_loader import WeatherDataLoader
        
        # Create weather loader
        weather_loader = WeatherDataLoader()
        print_test_result("Weather Loader creation", True)
        
        # Test weather fetching for today and next 2 days
        print("Fetching weather data...")
        today = date.today()
        weather_data = weather_loader.get_weather_for_trip(
            TEST_COORDINATES[0], TEST_COORDINATES[1],
            today, 3, TEST_CITY
        )
        
        print_test_result("Weather fetching", len(weather_data) > 0, f"Found {len(weather_data)} weather records")
        
        if weather_data:
            # Show sample weather
            sample_weather = weather_data[0]
            temp = sample_weather.temperature_avg
            condition = sample_weather.weather_condition
            print(f"    Sample: {sample_weather.date} - {temp}°C, {condition}")
            return True
        
        return False
        
    except Exception as e:
        print_test_result("Weather API", False, str(e))
        traceback.print_exc()
        return False


def test_data_validation():
    """Test data validation functionality"""
    print_test_header("Data Validation")
    
    try:
        from src.data_pipeline.data_validator import DataValidator
        from src.data_pipeline.osm_loader import OSMDataLoader
        from src.data_pipeline.weather_loader import WeatherDataLoader
        
        # Create validator
        validator = DataValidator()
        print_test_result("Validator creation", True)
        
        # Test API connections first
        print("Testing API connections...")
        api_results = validator.api_tester.run_all_api_tests()
        
        osm_success = api_results["osm"][0]
        weather_success = api_results["weather"][0]
        
        print_test_result("OSM API connection", osm_success)
        print_test_result("Weather API connection", weather_success)
        
        if osm_success and weather_success:
            # Test with small dataset
            print("Testing data validation with sample data...")
            
            # Get small POI sample
            osm_loader = OSMDataLoader()
            pois = osm_loader.fetch_pois(TEST_CITY, max_results=3)
            
            if pois:
                poi_validation = validator.validate_poi_data(pois)
                print_test_result("POI validation", poi_validation.is_valid, 
                                  f"Score: {poi_validation.score}/100")
            
            # Get small weather sample
            weather_loader = WeatherDataLoader()
            weather_data = weather_loader.get_weather_for_trip(
                TEST_COORDINATES[0], TEST_COORDINATES[1],
                date.today(), 2, TEST_CITY
            )
            
            if weather_data:
                weather_validation = validator.validate_weather_data(weather_data)
                print_test_result("Weather validation", weather_validation.is_valid,
                                  f"Score: {weather_validation.score}/100")
            
            return True
        
        return False
        
    except Exception as e:
        print_test_result("Data Validation", False, str(e))
        traceback.print_exc()
        return False


def test_full_integration():
    """Test full integration - complete data pipeline"""
    print_test_header("Full Integration Test")
    
    try:
        from src.data_pipeline.data_validator import DataValidator
        
        # Run complete validation
        validator = DataValidator()
        print("Running full validation (this may take 1-2 minutes)...")
        
        validation_report = validator.run_full_validation(TEST_CITY)
        
        overall_status = validation_report.get("overall_status", "unknown")
        print_test_result("Full integration", overall_status == "success", f"Status: {overall_status}")
        
        # Show summary
        if "api_tests" in validation_report:
            api_tests = validation_report["api_tests"]
            print(f"    OSM API: {'✅' if api_tests['osm'][0] else '❌'}")
            print(f"    Weather API: {'✅' if api_tests['weather'][0] else '❌'}")
        
        if "data_validation" in validation_report:
            data_val = validation_report["data_validation"]
            if "poi" in data_val:
                poi_score = data_val["poi"]["validation_result"].score
                print(f"    POI Quality: {poi_score:.1f}/100")
            if "weather" in data_val:
                weather_score = data_val["weather"]["validation_result"].score
                print(f"    Weather Quality: {weather_score:.1f}/100")
        
        return overall_status == "success"
        
    except Exception as e:
        print_test_result("Full Integration", False, str(e))
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    print_test_header("Dependency Check")
    
    required_packages = [
        "requests", "pandas", "numpy", "scikit-learn", 
        "python-dotenv", "pydantic", "psutil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle special package name mappings
            import_name = package.replace("-", "_")
            if package == "python-dotenv":
                import_name = "dotenv"
            elif package == "scikit-learn":
                import_name = "sklearn"
            
            __import__(import_name)
            print_test_result(f"Package {package}", True)
        except ImportError:
            print_test_result(f"Package {package}", False, "Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def main():
    """Run all Phase 1 tests"""
    print("🚀 Phase 1 Testing Script")
    print(f"Testing trip planner data pipeline for: {TEST_CITY}")
    print(f"Coordinates: {TEST_COORDINATES}")
    
    # Test results
    results = {}
    
    # 1. Check dependencies
    results["dependencies"] = check_dependencies()
    
    # 2. Test imports
    results["imports"] = test_imports()
    
    # 3. Test configuration
    results["configuration"] = test_configuration()
    
    # 4. Test utilities
    results["utilities"] = test_utilities()
    
    # 5. Test OSM API
    results["osm_api"] = test_osm_api()
    
    # 6. Test Weather API
    results["weather_api"] = test_weather_api()
    
    # 7. Test data validation
    results["data_validation"] = test_data_validation()
    
    # 8. Test full integration
    results["full_integration"] = test_full_integration()
    
    # Final summary
    print_test_header("FINAL RESULTS")
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print(f"\n📈 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Phase 1 is ready for Phase 2.")
        return True
    else:
        print("⚠️  Some tests failed. Fix issues before proceeding to Phase 2.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check API keys in .env file")
        print("- Verify internet connection")
        print("- Check file paths and module structure")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)