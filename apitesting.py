#!/usr/bin/env python3
"""
Simple API Ping Test
=====================

This script directly tests the WeatherAPI and OSM (Overpass) endpoints
to verify they are online and responding correctly.

It checks two things:
1.  Can we connect to the API server? (Network/DNS issue)
2.  Is our request authorized? (API key issue for WeatherAPI)

Usage:
    python api_ping_test.py
"""

import os
import requests
from requests.exceptions import RequestException

# --- Configuration ---

# We need to load the .env file to get the API key
# Make sure 'python-dotenv' is installed: pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file successfully.")
except ImportError:
    print("WARNING: 'python-dotenv' not found. Trying to read key from environment.")
    print("If the test fails, please install it: pip install python-dotenv")

# Get the API key from your environment variables
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

# Test data from your main script
TEST_COORDS = (17.6868, 83.2185)
TEST_CITY = "Visakhapatnam"

# API Endpoints
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"
OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# -----------------------

def print_header(title):
    print(f"\n{'='*50}")
    print(f"  Testing: {title}")
    print(f"{'='*50}")

def print_result(api_name, success, message):
    status = "✅ SUCCESS" if success else "❌ FAIL"
    print(f"{status} - {api_name}: {message}")

def test_weather_api():
    """Tests the WeatherAPI.com endpoint for connection and auth."""
    print_header("WeatherAPI.com")

    if not WEATHER_API_KEY:
        print_result("WeatherAPI", False, "WEATHER_API_KEY not found in .env file or environment.")
        return

    # Parameters for a simple "current weather" request
    params = {
        "key": WEATHER_API_KEY,
        "q": f"{TEST_COORDS[0]},{TEST_COORDS[1]}"
    }

    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)

        if response.status_code == 200:
            # Request was successful
            data = response.json()
            location = data.get('location', {}).get('name')
            print_result("WeatherAPI", True, f"Successfully fetched data for {location}.")
        
        elif response.status_code == 401:
            # Authentication error
            print_result("WeatherAPI", False, f"Status Code {response.status_code}. Your API key is invalid or has expired.")
        
        elif response.status_code == 403:
            # Forbidden (often a disabled key or quota issue)
            print_result("WeatherAPI", False, f"Status Code {response.status_code}. API key is valid but forbidden. Check your plan/quota.")
        
        else:
            # Other server error
            print_result("WeatherAPI", False, f"Received an error status code: {response.status_code}. Response: {response.text}")

    except RequestException as e:
        print_result("WeatherAPI", False, f"Could not connect to the API. Check your internet connection. Error: {e}")

def test_osm_api():
    """Tests the OpenStreetMap (Overpass) endpoint for connection."""
    print_header("OpenStreetMap (Overpass)")

    # A very simple query to find one node near the test coordinates
    overpass_query = f"""
    [out:json][timeout:10];
    node(around:1000, {TEST_COORDS[0]}, {TEST_COORDS[1]});
    out 1;
    """

    try:
        response = requests.post(OSM_OVERPASS_URL, data=overpass_query, timeout=10)

        if response.status_code == 200:
            # Request was successful
            print_result("OSM (Overpass)", True, "Successfully connected and received a valid response.")
        
        elif response.status_code == 429:
            # Rate limiting
            print_result("OSM (Overpass)", False, f"Status Code {response.status_code}. You are being rate-limited. Wait a bit and try again.")
        
        else:
            # Other server error
            print_result("OSM (Overpass)", False, f"Received an error status code: {response.status_code}. Response: {response.text}")

    except RequestException as e:
        print_result("OSM (Overpass)", False, f"Could not connect to the API. Check your internet connection. Error: {e}")

if __name__ == "__main__":
    print("🚀 Running Simple API Ping Test...")
    print("This will check connectivity and API keys.\n")
    
    test_weather_api()
    test_osm_api()
    
    print(f"\n{'='*50}")
    print("Ping test complete.")
    print("Check the '❌ FAIL' messages above to debug.")