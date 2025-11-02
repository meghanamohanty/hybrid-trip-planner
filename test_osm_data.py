# Save this as test_osm_data.py in your project's root folder

import logging
from src.data_pipeline.osm_loader import OSMDataLoader

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

TEST_CITY = "Visakhapatnam"

print(f"--- 🚀 Testing OSM Data Fetch for {TEST_CITY} ---")
print("This will fetch up to 1000 POIs to check for key categories...")

try:
    loader = OSMDataLoader()
    # Fetch a large number of POIs to get a good sample
    all_pois = loader.fetch_pois(TEST_CITY, max_results=1000)
    
    print(f"\nTotal POIs fetched and parsed: {len(all_pois)}")
    
    # --- Define what we are looking for ---
    # These are the (category, subcategory) tuples from your _determine_category function
    search_categories = {
        "Beaches": ("natural", "beach"),
        "Parks": ("leisure", "park"),
        "Museums": ("tourism", "museum"),
        "Temples": ("culture", "place_of_worship"),
        "Viewpoints": ("tourism", "viewpoint")
    }
    
    found_pois = {key: [] for key in search_categories}
    
    for poi in all_pois:
        for key, (cat, subcat) in search_categories.items():
            if poi.category == cat and poi.subcategory == subcat:
                found_pois[key].append(poi)
                
    print("\n--- 📊 Summary of Found POIs ---")
    
    if not any(found_pois.values()):
        print("\n❌ CRITICAL: No beaches, parks, museums, or temples were found!")
        print("This means the problem is in your osm_loader.py (query or _determine_category).")
    
    for key, pois in found_pois.items():
        print(f"\n--- {key.upper()} ({len(pois)} found) ---")
        if not pois:
            print(f"No POIs found matching category='{search_categories[key][0]}' and subcategory='{search_categories[key][1]}'")
        else:
            # Print top 5 found
            for i, poi in enumerate(pois[:5]): 
                print(f"  {i+1}. {poi.name} (Category: {poi.category}, SubCat: {poi.subcategory}, Outdoor: {poi.outdoor})")

except Exception as e:
    print(f"\n--- 💔 Test Failed ---")
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

print("\n--- ✅ Test Complete ---")