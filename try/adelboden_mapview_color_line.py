import pandas as pd
import numpy as np
import contextily as ctx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Load your data (replace 'file_path' with your actual file path)
data = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_Course_and_Athlete_Times_by_Gate.csv', delimiter=';')

# Filter data for Adelboden Run 2 only and drop rows with missing Longitude data
adelboden_data = data[(data['Venue'] == 'Adelboden') & (data['Run'] == 2)].dropna(subset=["Longitude (°)"])

# Sort by gate order
adelboden_data_sorted = adelboden_data.sort_values(by="Gate")

# Define the column to use for coloring ('relative_time_difference' or 'time_difference')
color_column = "relative_time_difference"  # Change this to "time_difference" if desired

# Extract relevant columns
latitude = adelboden_data_sorted["Latitude (°)"]
longitude = adelboden_data_sorted["Longitude (°)"]
altitude = adelboden_data_sorted["Altitude (m)"]
color_values = adelboden_data_sorted[color_column]

# Convert latitude and longitude to geospatial points and create GeoDataFrame
geometry = [Point(xy) for xy in zip(longitude, latitude)]
gdf = gpd.GeoDataFrame(adelboden_data_sorted, geometry=geometry)

# Set the coordinate reference system to WGS84 (lat/lon) and transform to Web Mercator for map plotting
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

# Normalize the color scale for chosen column
norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
colormap = plt.cm.RdYlGn  # Red to Yellow to Green colormap

# Convert points to (x, y) tuples
points = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))

# Create line segments between points
segments = np.array([points[i:i + 2] for i in range(len(points) - 1)])

# Create a LineCollection from the segments, colored by the chosen column values
line_collection = LineCollection(segments, cmap=colormap, norm=norm, linewidth=2)
line_collection.set_array(color_values.values[:-1])  # Apply color gradient based on values

# Plot the map view with a satellite image background
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the line collection
ax.add_collection(line_collection)

# Plot the points on top
gdf.plot(
    ax=ax,
    marker='o',
    column=color_column,
    cmap=colormap,
    markersize=10,
    legend=True,
    legend_kwds={'label': f"{color_column.replace('_', ' ').title()}"},
    norm=norm
)

# Calculate a buffer around the course to zoom out the view
buffer_factor = 0.05  # Adjust this factor to increase or decrease the zoom level
minx, miny, maxx, maxy = gdf.total_bounds
x_buffer = (maxx - minx) * buffer_factor
y_buffer = (maxy - miny) * buffer_factor

# Set the extent to include the buffer for zooming out
ax.set_xlim(minx - x_buffer, maxx + x_buffer)
ax.set_ylim(miny - y_buffer, maxy + y_buffer)

# Add contextily basemap (satellite map)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title(f"Adelboden Slalom Course (Run 2) with {color_column.replace('_', ' ').title()} Color Gradient")
plt.savefig('adelboden_map_line.png')
plt.show()
