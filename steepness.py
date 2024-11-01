import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data (replace 'file_path' with the actual path to your file)
data = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Course_Slalom.csv', delimiter=';')

# Filter data for Lenzerheide
lenzerheide_data = data[data['Venue'] == 'Lenzerheide']

# Sort by gate order to ensure correct sequence
lenzerheide_data_sorted = lenzerheide_data.sort_values(by="Gate (Nr)")

# Extract relevant columns
gate_gate_distance_lenzerheide = lenzerheide_data_sorted["Gate-Gate Distance (m)"]
steepness_lenzerheide = lenzerheide_data_sorted["Steepness [°]"]

# Calculate relative altitude changes based on steepness
relative_elevation_lenzerheide = [0]  # Starting point is 0 (relative)

# Calculate cumulative elevation changes
for i in range(len(gate_gate_distance_lenzerheide)):
    # Convert steepness to radians for trigonometric calculation
    slope_radians = np.radians(steepness_lenzerheide.iloc[i])
    # Calculate vertical change based on distance and slope angle
    vertical_change = gate_gate_distance_lenzerheide.iloc[i] * np.sin(slope_radians)
    # Append the calculated relative elevation change
    relative_elevation_lenzerheide.append(relative_elevation_lenzerheide[-1] + vertical_change)

# Calculate cumulative distance
cumulative_distances_lenzerheide = [0] + list(gate_gate_distance_lenzerheide.cumsum())

# Plot the relative elevation profile
plt.figure(figsize=(10, 6))
plt.plot(cumulative_distances_lenzerheide, relative_elevation_lenzerheide, marker='o', linestyle='-', color='blue', label='Relative Elevation')

# Label the plot
plt.xlabel('Cumulative Distance (m)')
plt.ylabel('Relative Elevation (m)')
plt.title('Relative Elevation Profile of Lenzerheide Slalom Course by Cumulative Distance')
plt.grid(True)


# Save the plot as a PNG file
plt.savefig('lenzerheide_relative_elevation.png')

plt.show()
