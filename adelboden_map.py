import pandas as pd
import numpy as np
import contextily as ctx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# Load your data (replace 'file_path' with your actual file path)
data = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_Course_and_Athlete_Times_by_Gate.csv', delimiter=';')

# Filter data for Adelboden Run 2 only and drop rows with missing Longitude data
adelboden_data = data[(data['Venue'] == 'Adelboden') & (data['Run'] == 2)].dropna(subset=["Longitude (°)"])

# Sort by gate order
adelboden_data_sorted = adelboden_data.sort_values(by="Gate")

# Extract relevant columns
latitude = adelboden_data_sorted["Latitude (°)"]
longitude = adelboden_data_sorted["Longitude (°)"]
altitude = adelboden_data_sorted["Altitude (m)"]
relative_time_diff = adelboden_data_sorted["relative_time_difference"]

# Convert latitude and longitude to geospatial points and create GeoDataFrame
geometry = [Point(xy) for xy in zip(longitude, latitude)]
gdf = gpd.GeoDataFrame(adelboden_data_sorted, geometry=geometry)

# Set the coordinate reference system to WGS84 (lat/lon) and transform to Web Mercator for map plotting
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

# Calculate a buffer around the course to zoom out the view
buffer_factor = 0.05  # Adjust this factor to increase or decrease the zoom level
minx, miny, maxx, maxy = gdf.total_bounds
x_buffer = (maxx - minx) * buffer_factor
y_buffer = (maxy - miny) * buffer_factor

# Normalize the color scale for relative_time_difference
norm = Normalize(vmin=relative_time_diff.min(), vmax=relative_time_diff.max())
colormap = plt.cm.RdYlGn  # Red to Yellow to Green colormap

# Plot the map view with a satellite image background
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(
    ax=ax,
    marker='o',
    column="relative_time_difference",
    cmap=colormap,
    markersize=50,
    legend=True,
    legend_kwds={'label': "Relative Time Difference (%)"},
    norm=norm
)

# Set the extent to include the buffer for zooming out
ax.set_xlim(minx - x_buffer, maxx + x_buffer)
ax.set_ylim(miny - y_buffer, maxy + y_buffer)

# Add contextily basemap (satellite map)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title('Adelboden Slalom Course (Run 2) with Relative Time Difference Color Gradient')
plt.show()
