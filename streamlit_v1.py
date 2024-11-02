import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import streamlit as st

# Load your data
data = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_Course_and_Athlete_Times_by_Gate.csv', delimiter=';')

# Define the plotting function
def plot_relative_elevation_profile(data, venue_name, run_number):
    # Filter data for the specified venue and run
    venue_data = data[(data['Venue'] == venue_name) & (data['Run'] == run_number)].dropna(subset=["Gate-Gate Distance (m)", "Steepness [°]", "relative_time_difference"])
    
    # Sort by gate order to ensure correct sequence
    venue_data_sorted = venue_data.sort_values(by="Gate")
    
    # Extract relevant columns
    gate_gate_distance = venue_data_sorted["Gate-Gate Distance (m)"]
    steepness = venue_data_sorted["Steepness [°]"]
    relative_time_diff = venue_data_sorted["relative_time_difference"]
    
    # Calculate relative altitude changes based on steepness
    relative_elevation = [0]  # Starting point is 0 (relative)
    
    # Calculate cumulative elevation changes
    for i in range(len(gate_gate_distance)):
        # Convert steepness to radians for trigonometric calculation
        slope_radians = np.radians(steepness.iloc[i])
        # Calculate vertical change based on distance and slope angle
        vertical_change = gate_gate_distance.iloc[i] * np.sin(slope_radians)
        # Append the calculated relative elevation change
        relative_elevation.append(relative_elevation[-1] + vertical_change)
    
    # Calculate cumulative distance
    cumulative_distances = [0] + list(gate_gate_distance.cumsum())
    
    # Normalize the color scale for relative_time_difference
    norm = Normalize(vmin=relative_time_diff.min(), vmax=relative_time_diff.max())
    colormap = plt.cm.RdYlGn  # Red to Yellow to Green colormap
    
    # Plot the relative elevation profile
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each point with color based on relative_time_difference
    for x, y, color_val in zip(cumulative_distances, relative_elevation, relative_time_diff):
        ax.plot(x, y, marker='o', color=colormap(norm(color_val)), markersize=8)
    
    # Create a colorbar and add it to the plot
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Only needed for ScalarMappable to work with colorbar
    plt.colorbar(sm, ax=ax, label="Relative Time Difference (%)")
    
    # Label the plot
    ax.set_xlabel('Cumulative Distance (m)')
    ax.set_ylabel('Relative Elevation (m)')
    ax.set_title(f'Relative Elevation Profile of {venue_name} Slalom Course (Run {run_number}) by Cumulative Distance')
    ax.grid(True)
    
    return fig

# Streamlit app
st.title("Slalom Course Relative Elevation Profile")

# Dropdown for selecting venue
venues = data['Venue'].unique()
selected_venue = st.selectbox("Select a Venue", venues)

# Dropdown for selecting run
run_number = st.selectbox("Select Run Number", [1, 2])

# Plot and display the relative elevation profile
if st.button("Generate Plot"):
    fig = plot_relative_elevation_profile(data, selected_venue, run_number)
    st.pyplot(fig)
