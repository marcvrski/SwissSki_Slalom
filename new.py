import ee
import geemap
import folium
import geopandas as gpd
from shapely.geometry import Point

# Initialize Earth Engine
ee.Initialize()

# Define the Adelboden location and date range for winter imagery
adelboden_coords = ee.Geometry.Point([7.5586, 46.4926])  # Approximate coordinates of Adelboden
start_date = '2024-01-01'
end_date = '2024-02-28'

# Filter Sentinel-2 data for this location and date range
sentinel_collection = (
    ee.ImageCollection('COPERNICUS/S2')
    .filterBounds(adelboden_coords)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Filter for clear images
)

# Get median image from the collection
winter_image = sentinel_collection.median().clip(adelboden_coords.buffer(20000))  # 20 km buffer around Adelboden

# Visualize the Sentinel-2 image with folium and geemap
Map = geemap.Map(center=[46.4926, 7.5586], zoom=12)
Map.addLayer(winter_image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}, 'Winter Sentinel-2')
Map.add_child(folium.LayerControl())

# Display the map with folium
Map.add_basemap('HYBRID')
Map.show()
