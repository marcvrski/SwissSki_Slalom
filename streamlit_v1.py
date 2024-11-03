import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Load your data
#data = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_Course_and_Athlete_Times_by_Gate.csv', delimiter=';')

# Define the plotting function with Plotly
def plot_relative_elevation_profile(data, venue_name, run_number):
    # Filter data for the specified venue and run
    venue_data = data[(data['Venue'] == venue_name) & (data['Run'] == run_number)].dropna(subset=["Gate-Gate Distance (m)", "Steepness [°]", "relative_time_difference", "ref_time"])

    # Sort by gate order to ensure correct sequence
    venue_data_sorted = venue_data.sort_values(by="Gate")

    # Extract relevant columns
    gate_gate_distance = venue_data_sorted["Gate-Gate Distance (m)"]
    steepness = venue_data_sorted["Steepness [°]"]
    relative_time_diff = venue_data_sorted["relative_time_difference"]
    abs_time_diff = venue_data_sorted["time_difference"]
    ref_time = venue_data_sorted["ref_time"]
    gate = venue_data_sorted["Gate"]
    offset = venue_data_sorted["Offset [m]"]    
    turning_angle = venue_data_sorted["Turning Angle [°]"] 


# Display additional information in Streamlit
    st.markdown(f"**Date:** {venue_data_sorted['Date'].iloc[0]}")
    st.metric(label="Athlete Name (Rev)", value=venue_data_sorted['athlete_name_ref'].iloc[0])
    st.metric(label="Athlete Name (Compare)", value=venue_data_sorted['athlete_name_athlete_2'].iloc[0])

  # Display the dataset in Streamlit
    st.dataframe(venue_data_sorted)

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
    
    # Normalize relative_time_difference to 0-1 scale for color gradient
    normalized_diff = (relative_time_diff - relative_time_diff.min()) / (relative_time_diff.max() - relative_time_diff.min())
    colors = normalized_diff.apply(lambda x: f'rgba({int((1 - x) * 255)}, {int(x * 255)}, 0, 0.8)')

    # Create Plotly scatter plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cumulative_distances,
            y=relative_elevation,
            mode="markers+lines",
            marker=dict(color=colors, size=10),
            line=dict(color="lightgray", width=1),
            text=[f"Gate: {g}<br>Absolute Time Difference: {t:.2f} s<br>Turning Angle: {ta:.2f}°<br>Offset: {o:.2f} m<br>Steepness: {s:.2f}°" for g, t, ta, o, s in zip(gate, abs_time_diff, turning_angle, offset, steepness)],
            hoverinfo="text"
        )
    )
    
    # Customize the layout
    fig.update_layout(
        title=f"Relative Elevation Profile of {venue_name} Slalom Course (Run {run_number})",
        xaxis_title="Cumulative Distance (m)",
        yaxis_title="Relative Elevation (m)",
        template="plotly_white",
        showlegend=False
    )
    
    return fig

# Streamlit app
st.title("Slalom Course Relative Elevation Profile")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, delimiter=';')
    st.write("filename:", uploaded_file.name)
    st.write(data.head())

    # Dropdown for selecting venue
    venues = data['Venue'].unique()
    selected_venue = st.selectbox("Select a Venue", venues)

    # Dropdown for selecting run
    run_number = st.selectbox("Select Run Number", [1, 2])

    # Plot and display the relative elevation profile
    if st.button("Analyse"):
        fig = plot_relative_elevation_profile(data, selected_venue, run_number)
        st.plotly_chart(fig)

    st.snow()