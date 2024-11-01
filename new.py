import pandas as pd
import numpy as np
import contextily as ctx
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Load your data (replace 'file_path' with your actual file path)
data = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Course_Slalom.csv', delimiter=';')

# Filter data for Lenzerheide only
lenzerheide_data = data[data['Venue'] == 'Lenzerheide']

# Sort by gate order
lenzerheide_data_sorted = lenzerheide_data.sort_values(by="Gate (Nr)")

# Extract relevant columns
latitude = lenzerheide_data_sorted["Latitude (°)"]
longitude = lenzerheide_data_sorted["Longitude (°)"]
altitude = lenzerheide_data_sorted["Altitude (m)"]

# Function to calculate 3D distance between points
def calculate_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    horizontal_distance = geodesic((lat1, lon1), (lat2, lon2)).meters
    vertical_distance = abs(alt2 - alt1)
    return np.sqrt(horizontal_distance**2 + vertical_distance**2)

# Calculate distances between consecutive gates
distances = []
for i in range(1, len(latitude)):
    lat1, lon1, alt1 = latitude.iloc[i-1], longitude.iloc[i-1], altitude.iloc[i-1]
    lat2, lon2, alt2 = latitude.iloc[i], longitude.iloc[i], altitude.iloc[i]
    distances.append(calculate_distance(lat1, lon1, alt1, lat2, lon2, alt2))

# Calculate cumulative distances
cumulative_distances = [0] + list(np.cumsum(distances))

# Plot altitude profile with cumulative distance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_distances, altitude, marker='o', linestyle='-', color='blue')
plt.xlabel('Cumulative Distance (m)')
plt.ylabel('Altitude (m)')
plt.title('Altitude Profile of the Lenzerheide Slalom Course by Cumulative Distance')
plt.grid(True)
plt.show()

# Convert latitude and longitude to geospatial points and create GeoDataFrame
geometry = [Point(xy) for xy in zip(longitude, latitude)]
gdf = gpd.GeoDataFrame(lenzerheide_data_sorted, geometry=geometry)

# Set the coordinate reference system to WGS84 (lat/lon) and transform to Web Mercator for map plotting
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

# Plot the map view with a satellite image background
fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(ax=ax, marker='o', color='green', linestyle='-', label='Slalom Course Path')

# Add contextily basemap (satellite map)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title('Map View of the Lenzerheide Slalom Course')
plt.legend()
# Save the plot as a PNG file
plt.savefig('lenzerheide_map.png')
plt.show()
