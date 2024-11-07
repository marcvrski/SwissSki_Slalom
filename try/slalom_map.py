import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# Load your data (replace 'file_path' with your actual file path)
data = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Course_Slalom.csv', delimiter=';')

print(data.head())  

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

import contextily as ctx

# Convert latitude and longitude to the Web Mercator projection (EPSG:3857) for contextily compatibility
lon_lat = pd.DataFrame({'Longitude': longitude, 'Latitude': latitude})
lon_lat = lon_lat.to_crs(epsg=3857)

# Plot the map view with a satellite image or map background
plt.figure(figsize=(10, 6))
plt.plot(lon_lat['Longitude'], lon_lat['Latitude'], marker='o', linestyle='-', color='green')
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title('Map View of the Lenzerheide Slalom Course')

# Add contextily basemap
ctx.add_basemap(plt.gca(), source=ctx.providers.Esri.WorldImagery)

plt.grid(True)
plt.show()
