#!/usr/bin/env python3
"""
Phase 2 Testing Script
=====================

Comprehensive test to verify all Phase 2 ML engine components work correctly.
Tests scoring, clustering, routing, budgeting, and weather adjustments.

Usage:
    python test_phase2.py

Author: Hybrid Trip Planner Team
"""

import sys
import os
from datetime import date, timedelta

# Test configuration
TEST_CITY = "Visakhapatnam"
TEST_COORDINATES = (17.6868, 83.2185)
TEST_BUDGET = 20000  # INR
TEST_DAYS = 3


def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")


def print_test_result(test_name: str, success: bool, message: str = ""):
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"      {message}")


def test_imports():
    """Test if all Phase 2 modules can be imported"""
    print_test_header("Phase 2 Module Imports")
    
    import_tests = [
        ("POIScorer", "from src.ml_engine.poi_scorer import POIScorer, ScoredPOI"),
        ("TripClusterer", "from src.ml_engine.trip_clusterer import TripClusterer, DayCluster"),
        ("RouteOptimizer", "from src.ml_engine.route_optimizer import RouteOptimizer, OptimizedRoute"),
        ("BudgetEstimator", "from src.ml_engine.budget_estimator import BudgetEstimator, TripBudget"),
        ("WeatherAdjuster", "from src.ml_engine.weather_adjuster import WeatherAdjuster, WeatherAdjustedDay"),
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
    
    return all(results.values())


def test_poi_scoring():
    """Test POI scoring functionality"""
    print_test_header("POI Scoring")
    
    try:
        from src.ml_engine.poi_scorer import POIScorer
        from src.data_pipeline.osm_loader import OSMDataLoader
        from src.data_pipeline.weather_loader import WeatherDataLoader
        
        # Initialize scorer
        scorer = POIScorer()
        print_test_result("POIScorer initialization", True)
        
        # Fetch test POIs
        print("Fetching test POIs...")
        osm_loader = OSMDataLoader()
        pois = osm_loader.fetch_pois(TEST_CITY, max_results=20)
        
        if not pois:
            print_test_result("POI fetching", False, "No POIs fetched")
            return False
        
        print_test_result("POI fetching", True, f"Fetched {len(pois)} POIs")
        
        # Get weather data
        weather_loader = WeatherDataLoader()
        weather_data = weather_loader.get_weather_for_trip(
            TEST_COORDINATES[0], TEST_COORDINATES[1],
            date.today(), 1, TEST_CITY
        )
        weather = weather_data[0] if weather_data else None
        
        # Score POIs
        print("Scoring POIs...")
        scored_pois = scorer.score_pois(
            pois,
            TEST_COORDINATES[0], TEST_COORDINATES[1],
            user_interests=['tourism', 'historic', 'leisure', 'natural', 'food', 'culture'],
            weather_data=weather,
            budget_per_poi=500
        )
        

        if not scored_pois:
            print_test_result("POI scoring", False, "No POIs scored")
            return False
        
        print_test_result("POI scoring", True, f"Scored {len(scored_pois)} POIs")
        
        # Verify score ranges
        all_scores_valid = all(0 <= poi.total_score <= 1 for poi in scored_pois)
        print_test_result("Score range validation (0-1)", all_scores_valid)
        
        # Show top 3 POIs
        print("\n      Top 3 scored POIs:")
        for i, poi in enumerate(scored_pois[:3], 1):
            print(f"      {i}. {poi.poi.name} - Score: {poi.total_score:.3f}")
        
        # Get statistics
        stats = scorer.get_score_statistics(scored_pois)
        print(f"      Average score: {stats['avg_score']:.3f}")
        print(f"      POIs above 0.5: {stats['scores_above_0.5']}")
        
        return True
        
    except Exception as e:
        print_test_result("POI Scoring", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_clustering():
    """Test trip clustering functionality"""
    print_test_header("Trip Clustering")
    
    try:
        from src.ml_engine.poi_scorer import POIScorer
        from src.ml_engine.trip_clusterer import TripClusterer
        from src.data_pipeline.osm_loader import OSMDataLoader
        
        # Get and score POIs
        osm_loader = OSMDataLoader()
        pois = osm_loader.fetch_pois(TEST_CITY, max_results=30)
        
        scorer = POIScorer()
        scored_pois = scorer.score_pois(
            pois, TEST_COORDINATES[0], TEST_COORDINATES[1]
        )
        
        # Initialize clusterer
        clusterer = TripClusterer()
        print_test_result("TripClusterer initialization", True)
        
        # Cluster POIs
        print(f"Clustering POIs into {TEST_DAYS} days...")
        day_clusters = clusterer.cluster_pois_by_days(
            scored_pois, TEST_DAYS
        )
        
        if not day_clusters:
            print_test_result("Clustering", False, "No clusters created")
            return False
        
        print_test_result("Clustering", True, f"Created {len(day_clusters)} clusters")
        
        # Verify constraints
        for cluster in day_clusters:
            poi_count = len(cluster.pois)
            radius = cluster.radius_km
            
            print(f"\n      Day {cluster.day_number}:")
            print(f"         POIs: {poi_count}")
            print(f"         Radius: {radius:.2f} km")
            print(f"         Center: ({cluster.center_lat:.4f}, {cluster.center_lon:.4f})")
            
            # Check constraints
            within_poi_range = 3 <= poi_count <= 8
            within_radius = radius <= 5.0
            
            if not within_poi_range:
                print(f"         ⚠️ POI count outside range (3-8)")
            if not within_radius:
                print(f"         ⚠️ Radius exceeds 5km limit")
        
        # Get statistics
        stats = clusterer.get_cluster_statistics(day_clusters)
        print(f"\n      Statistics:")
        print(f"         Total POIs: {stats['total_pois']}")
        print(f"         Avg POIs/day: {stats['pois_per_day']['avg']:.1f}")
        print(f"         Avg radius: {stats['cluster_radius_km']['avg']:.2f} km")
        
        return True
        
    except Exception as e:
        print_test_result("Clustering", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_routing():
    """Test route optimization functionality"""
    print_test_header("Route Optimization")
    
    try:
        from src.ml_engine.poi_scorer import POIScorer
        from src.ml_engine.trip_clusterer import TripClusterer
        from src.ml_engine.route_optimizer import RouteOptimizer
        from src.data_pipeline.osm_loader import OSMDataLoader
        
        # Get, score, and cluster POIs
        osm_loader = OSMDataLoader()
        pois = osm_loader.fetch_pois(TEST_CITY, max_results=25)
        
        scorer = POIScorer()
        scored_pois = scorer.score_pois(pois, TEST_COORDINATES[0], TEST_COORDINATES[1])
        
        clusterer = TripClusterer()
        day_clusters = clusterer.cluster_pois_by_days(scored_pois, TEST_DAYS)
        
        # Initialize optimizer
        optimizer = RouteOptimizer()
        print_test_result("RouteOptimizer initialization", True)
        
        # Optimize routes
        print("Optimizing routes...")
        routes = optimizer.optimize_all_routes(day_clusters, travel_mode='walking')
        
        if not routes:
            print_test_result("Route optimization", False, "No routes created")
            return False
        
        print_test_result("Route optimization", True, f"Optimized {len(routes)} routes")
        
        # Display route details
        for route in routes:
            print(f"\n      Day {route.day_number}:")
            print(f"         Stops: {len(route.stops)}")
            print(f"         Distance: {route.total_distance_km} km")
            print(f"         Travel time: {route.total_travel_time_minutes} min")
            print(f"         Start: {route.start_time}")
            print(f"         End: {route.end_time}")
            print(f"         Feasible: {'✓' if route.is_feasible else '✗'}")
            
            # Show first 3 stops
            if route.stops:
                print(f"         First stops:")
                for stop in route.stops[:3]:
                    print(f"            {stop.arrival_time} - {stop.poi.name}")
        
        # Get statistics
        stats = optimizer.get_route_statistics(routes)
        print(f"\n      Overall Statistics:")
        print(f"         Total distance: {stats['total_distance_km']} km")
        print(f"         Total travel time: {stats['total_travel_time_minutes']} min")
        print(f"         All feasible: {'✓' if stats['all_routes_feasible'] else '✗'}")
        
        return True
        
    except Exception as e:
        print_test_result("Route Optimization", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_budgeting():
    """Test budget estimation functionality"""
    print_test_header("Budget Estimation")
    
    try:
        from src.ml_engine.poi_scorer import POIScorer
        from src.ml_engine.trip_clusterer import TripClusterer
        from src.ml_engine.route_optimizer import RouteOptimizer
        from src.ml_engine.budget_estimator import BudgetEstimator
        from src.data_pipeline.osm_loader import OSMDataLoader
        
        # Get complete trip plan
        osm_loader = OSMDataLoader()
        pois = osm_loader.fetch_pois(TEST_CITY, max_results=25)
        
        scorer = POIScorer()
        scored_pois = scorer.score_pois(pois, TEST_COORDINATES[0], TEST_COORDINATES[1])
        
        clusterer = TripClusterer()
        day_clusters = clusterer.cluster_pois_by_days(scored_pois, TEST_DAYS)
        
        optimizer = RouteOptimizer()
        routes = optimizer.optimize_all_routes(day_clusters)
        
        # Initialize estimator
        estimator = BudgetEstimator()
        print_test_result("BudgetEstimator initialization", True)
        
        # Estimate budget
        print(f"Estimating budget for ₹{TEST_BUDGET}...")
        trip_budget = estimator.estimate_trip_budget(
            routes, TEST_BUDGET, TEST_DAYS
        )
        
        print_test_result("Budget estimation", True)
        
        # Display budget breakdown
        costs = trip_budget.estimated_costs
        print(f"\n      Budget Breakdown:")
        print(f"         Total budget: ₹{trip_budget.total_budget:,.0f}")
        print(f"         Estimated cost: ₹{costs.total:,.0f}")
        print(f"         Hotels: ₹{costs.hotels:,.0f}")
        print(f"         Food: ₹{costs.food:,.0f}")
        print(f"         Transport: ₹{costs.transport:,.0f}")
        print(f"         Tickets: ₹{costs.tickets:,.0f}")
        print(f"         Variance: ₹{trip_budget.budget_variance:,.0f}")
        print(f"         Feasible: {'✓' if trip_budget.is_feasible else '✗'}")
        
        # Show warnings
        if trip_budget.warnings:
            print(f"\n      Warnings:")
            for warning in trip_budget.warnings:
                print(f"         ⚠️ {warning}")
        
        # Get statistics
        stats = estimator.get_budget_statistics(trip_budget)
        print(f"\n      Budget Utilization: {stats['budget_utilization_pct']:.1f}%")
        
        return True
        
    except Exception as e:
        print_test_result("Budget Estimation", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_weather_adjustment():
    """Test weather adjustment functionality"""
    print_test_header("Weather Adjustment")
    
    try:
        from src.ml_engine.poi_scorer import POIScorer
        from src.ml_engine.trip_clusterer import TripClusterer
        from src.ml_engine.weather_adjuster import WeatherAdjuster
        from src.data_pipeline.osm_loader import OSMDataLoader
        from src.data_pipeline.weather_loader import WeatherDataLoader
        
        # Get POIs and weather
        osm_loader = OSMDataLoader()
        pois = osm_loader.fetch_pois(TEST_CITY, max_results=25)
        
        scorer = POIScorer()
        scored_pois = scorer.score_pois(pois, TEST_COORDINATES[0], TEST_COORDINATES[1])
        
        clusterer = TripClusterer()
        day_clusters = clusterer.cluster_pois_by_days(scored_pois, TEST_DAYS)
        
        weather_loader = WeatherDataLoader()
        weather_data = weather_loader.get_weather_for_trip(
            TEST_COORDINATES[0], TEST_COORDINATES[1],
            date.today(), TEST_DAYS, TEST_CITY
        )
        
        # Initialize adjuster
        adjuster = WeatherAdjuster()
        print_test_result("WeatherAdjuster initialization", True)
        
        # Adjust for weather
        print("Adjusting itinerary for weather...")
        adjusted_days = adjuster.adjust_day_clusters(
            day_clusters, weather_data, date.today()
        )
        
        if not adjusted_days:
            print_test_result("Weather adjustment", False, "No adjusted days")
            return False
        
        print_test_result("Weather adjustment", True, f"Adjusted {len(adjusted_days)} days")
        
        # Display adjustments
        for adj_day in adjusted_days:
            weather = adj_day.weather_data
            print(f"\n      Day {adj_day.day_number} ({adj_day.date}):")
            print(f"         Weather: {weather.weather_condition or 'N/A'}")
            print(f"         Temp: {weather.temperature_avg or 0:.1f}°C")
            print(f"         Rain: {weather.precipitation or 0:.1f}mm")
            print(f"         Suitability: {adj_day.weather_suitability:.2f}")
            print(f"         Indoor POIs: {adj_day.indoor_pois_count}")
            print(f"         Outdoor POIs: {adj_day.outdoor_pois_count}")
            
            # Show recommendations
            if adj_day.recommendations:
                print(f"         Recommendations:")
                for rec in adj_day.recommendations[:2]:
                    print(f"            • {rec}")
        
        # Get statistics
        stats = adjuster.get_weather_statistics(adjusted_days)
        print(f"\n      Weather Statistics:")
        print(f"         Avg penalty: {stats['avg_weather_penalty']:.3f}")
        print(f"         Days with poor weather: {stats['days_with_poor_weather']}")
        print(f"         Days with good weather: {stats['days_with_good_weather']}")
        
        return True
        
    except Exception as e:
        print_test_result("Weather Adjustment", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test complete ML pipeline integration with DETAILED PLACE LISTINGS"""
    print_test_header("Full ML Pipeline Integration + Detailed Itinerary")
    
    try:
        print("Running complete ML pipeline...")
        
        # All imports
        from src.data_pipeline.osm_loader import OSMDataLoader
        from src.data_pipeline.weather_loader import WeatherDataLoader
        from src.ml_engine.poi_scorer import POIScorer
        from src.ml_engine.trip_clusterer import TripClusterer
        from src.ml_engine.route_optimizer import RouteOptimizer
        from src.ml_engine.budget_estimator import BudgetEstimator
        from src.ml_engine.weather_adjuster import WeatherAdjuster
        
        # Step 1: Fetch data
        print("   1. Fetching POIs and weather...")
        osm_loader = OSMDataLoader()
        weather_loader = WeatherDataLoader()
        
        pois = osm_loader.fetch_pois(TEST_CITY, max_results=30)
        weather_data = weather_loader.get_weather_for_trip(
            TEST_COORDINATES[0], TEST_COORDINATES[1],
            date.today(), TEST_DAYS, TEST_CITY
        )
        
        print(f"      ✓ Fetched {len(pois)} POIs")
        print(f"      ✓ Fetched {len(weather_data)} weather records")
        
        # Step 2: Score POIs
        print("   2. Scoring POIs...")
        scorer = POIScorer()
        scored_pois = scorer.score_pois(
            pois,
            TEST_COORDINATES[0], TEST_COORDINATES[1],
            user_interests=['tourism', 'historic', 'leisure', 'natural', 'food', 'culture'],
            weather_data=weather_data[0] if weather_data else None
        )
        print(f"      ✓ Scored {len(scored_pois)} POIs")
        
        # Step 3: Cluster into days
        print("   3. Clustering into days...")
        clusterer = TripClusterer()
        day_clusters = clusterer.cluster_pois_by_days(scored_pois, TEST_DAYS)
        print(f"      ✓ Created {len(day_clusters)} day clusters")
        
        # Step 4: Optimize routes
        print("   4. Optimizing routes...")
        optimizer = RouteOptimizer()
        routes = optimizer.optimize_all_routes(day_clusters)
        print(f"      ✓ Optimized {len(routes)} routes")
        
        # Step 5: Estimate budget
        print("   5. Estimating budget...")
        estimator = BudgetEstimator()
        trip_budget = estimator.estimate_trip_budget(routes, TEST_BUDGET, TEST_DAYS)
        print(f"      ✓ Budget: ₹{trip_budget.estimated_costs.total:,.0f} / ₹{TEST_BUDGET:,.0f}")
        
        # Step 6: Adjust for weather
        print("   6. Adjusting for weather...")
        adjuster = WeatherAdjuster()
        adjusted_days = adjuster.adjust_day_clusters(
            day_clusters, weather_data, date.today()
        )
        print(f"      ✓ Adjusted {len(adjusted_days)} days")
        
        print_test_result("Full Integration", True, "Complete ML pipeline executed successfully")
        
        # ========== DETAILED DAILY ITINERARY ==========
        print("\n" + "="*80)
        print("  📅 DETAILED DAILY ITINERARY - ALL PLACES")
        print("="*80)
        
        for route in routes:
            day_num = route.day_number
            adj_day = adjusted_days[day_num - 1] if day_num <= len(adjusted_days) else None
            
            print(f"\n{'='*80}")
            print(f"  DAY {day_num}")
            if adj_day:
                print(f"  {adj_day.date.strftime('%A, %B %d, %Y')}")
            print(f"{'='*80}")
            
            # Weather
            if adj_day and adj_day.weather_data:
                w = adj_day.weather_data
                print(f"\n   ☀️ Weather: {w.temperature_avg:.1f}°C, {w.weather_condition or 'N/A'}, Rain: {w.precipitation or 0:.1f}mm")
                print(f"      Outdoor Suitability: {w.outdoor_suitability:.0%}")
            
            # Route summary
            print(f"\n   🗺️ Route: {len(route.stops)} places | {route.total_distance_km:.1f} km | {route.total_travel_time_minutes:.0f} min travel")
            print(f"      Time: {route.start_time} to {route.end_time}")
            
            # ALL PLACES FOR THIS DAY
            print(f"\n   📍 ALL PLACES TO VISIT:")
            print(f"  {'-'*78}")
            
            for idx, stop in enumerate(route.stops, 1):
                poi = stop.poi
                
                # Emoji by category
                emoji_map = {
                    "tourism": "🏛️", "historic": "🏰", "leisure": "🎭",
                    "food": "🍽️", "shopping": "🛍️", "natural": "🌳",
                    "culture": "🎨", "entertainment": "🎪"
                }
                emoji = emoji_map.get(poi.category, "📍")
                
                print(f"\n  {idx}. {emoji} {poi.name}")
                print(f"     ⏰ {stop.arrival_time} - {stop.departure_time} ({stop.duration_minutes} min)")
                print(f"     🏷️ {poi.category.title()} ({poi.subcategory or 'general'}) | {'🌤️ Outdoor' if poi.outdoor else '🏠 Indoor'}")
                print(f"     📍 {poi.latitude:.4f}°N, {poi.longitude:.4f}°E")
                
                if poi.address:
                    print(f"        {poi.address}")
                if poi.phone:
                    print(f"     📞 {poi.phone}")
                if poi.website:
                    print(f"     🌐 {poi.website}")
                if poi.opening_hours:
                    print(f"     🕐 Hours: {poi.opening_hours}")
                if poi.fee_required is not None:
                    fee_text = "Yes" if poi.fee_required else "Free"
                    print(f"     💳 Entry: {fee_text}")
                if poi.rating:
                    stars = "⭐" * int(poi.rating)
                    print(f"     {stars} {poi.rating:.1f}/5")
                
                if idx > 1:
                    print(f"     🚶 From previous: {stop.travel_from_previous_km:.2f} km ({stop.travel_time_minutes:.0f} min)")
            
            # Day cost
            day_budget_obj = None
            for db in trip_budget.daily_budgets:
                if db.day_number == day_num:
                    day_budget_obj = db
                    break
            
            if day_budget_obj:
                day_costs = day_budget_obj.costs
                print(f"\n   💵 Day {day_num} Budget: ₹{day_costs.total:,.0f}")
                print(f"      Hotels: ₹{day_costs.hotels:,.0f}")
                print(f"      Food: ₹{day_costs.food:,.0f}")
                print(f"      Transport: ₹{day_costs.transport:,.0f}")
                print(f"      Tickets: ₹{day_costs.tickets:,.0f}")
        
        # Final summary
        print("\n" + "="*80)
        print("  📊 TRIP SUMMARY")
        print("="*80)
        print(f"\n   ✨ {TEST_DAYS}-Day Trip to {TEST_CITY}")
        print(f"\n      📍 Total Places: {sum(len(r.stops) for r in routes)}")
        print(f"      🚶 Total Distance: {sum(r.total_distance_km for r in routes):.1f} km")
        print(f"      ⏱️ Total Travel Time: {sum(r.total_travel_time_minutes for r in routes):.0f} min")
        print(f"      💰 Total Cost: ₹{trip_budget.estimated_costs.total:,.0f}")
        print(f"      🏨 Hotels: ₹{trip_budget.estimated_costs.hotels:,.0f}")
        print(f"      🍽️ Food: ₹{trip_budget.estimated_costs.food:,.0f}")
        
        # Category breakdown
        all_pois = [stop.poi for route in routes for stop in route.stops]
        cat_count = {}
        for poi in all_pois:
            cat_count[poi.category] = cat_count.get(poi.category, 0) + 1
        
        print(f"\n      📊 Places by Category:")
        for cat, count in sorted(cat_count.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(all_pois) * 100
            print(f"         {cat.title()}: {count} ({pct:.1f}%)")
        
        outdoor_count = sum(1 for poi in all_pois if poi.outdoor)
        print(f"\n      🏠 Indoor: {len(all_pois) - outdoor_count} | 🌤️ Outdoor: {outdoor_count}")
        print(f"      {'✅ Within Budget' if trip_budget.is_feasible else '⚠️ Over Budget'}")
        print(f"      {'✅ All Days Feasible' if all(r.is_feasible for r in routes) else '⚠️ Tight Schedule'}")
        
        print("\n" + "="*80)
        
        return True
        
    except Exception as e:
        print_test_result("Full Integration", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2 tests"""
    print("🚀 Phase 2 ML Engine Testing")
    print(f"Testing ML recommendation engine for {TEST_CITY}")
    print(f"Configuration: {TEST_DAYS} days, ₹{TEST_BUDGET:,} budget\n")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    if not results['imports']:
        print("\n❌ Import tests failed. Fix imports before continuing.")
        return False
    
    # Test 2: POI Scoring
    results['scoring'] = test_poi_scoring()
    
    # Test 3: Clustering
    results['clustering'] = test_clustering()
    
    # Test 4: Routing
    results['routing'] = test_routing()
    
    # Test 5: Budgeting
    results['budgeting'] = test_budgeting()
    
    # Test 6: Weather Adjustment
    results['weather'] = test_weather_adjustment()
    
    # Test 7: Full Integration
    results['integration'] = test_full_integration()
    
    # Final Results
    print_test_header("FINAL RESULTS")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 tests passed! Ready for Phase 3 (GenAI Integration).")
        return True
    else:
        print("⚠️ Some tests failed. Review errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)