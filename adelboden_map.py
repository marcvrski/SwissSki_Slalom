import pandas as pd
import numpy as np
import contextily as ctx
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Load your data (replace 'file_path' with your actual file path)
data = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Course_Slalom.csv', delimiter=';')

# Filter data for Adelboden only and drop rows with missing Longitude data
adelboden_data = data[data['Venue'] == 'Adelboden'].dropna(subset=["Longitude (°)"])

# Sort by gate order
adelboden_data_sorted = adelboden_data.sort_values(by="Gate (Nr)")

# Extract relevant columns
latitude = adelboden_data_sorted["Latitude (°)"]
longitude = adelboden_data_sorted["Longitude (°)"]
altitude = adelboden_data_sorted["Altitude (m)"]

# Convert latitude and longitude to geospatial points and create GeoDataFrame
geometry = [Point(xy) for xy in zip(longitude, latitude)]
gdf = gpd.GeoDataFrame(adelboden_data_sorted, geometry=geometry)

# Set the coordinate reference system to WGS84 (lat/lon) and transform to Web Mercator for map plotting
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

# Calculate a buffer around the course to zoom out the view
buffer_factor = 0.8  # Adjust this factor to increase or decrease the zoom level
minx, miny, maxx, maxy = gdf.total_bounds
x_buffer = (maxx - minx) * buffer_factor
y_buffer = (maxy - miny) * buffer_factor

# Plot the map view with a satellite image background
fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size for a better view
gdf.plot(ax=ax, marker='o', color='green', linestyle='-', label='Slalom Course Path')

# Set the extent to include the buffer for zooming out
ax.set_xlim(minx - x_buffer, maxx + x_buffer)
ax.set_ylim(miny - y_buffer, maxy + y_buffer)

# Add contextily basemap (satellite map)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title('Zoomed-Out Map View of the Adelboden Slalom Course')
plt.legend()
plt.savefig('adelboden_map.png')

plt.show()
