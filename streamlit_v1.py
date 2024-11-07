import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

def summarize_data(data, venue, run):
    rev_athlete = data[data['Best'] == st.session_state['athlete_name_ref']]
    compare_athlete = data[(data['Best'] == st.session_state['athlete_name_athlete_2'])]
   
     #Course data
    st.subheader("Race and Course Details:")
    st.write("Gates :",rev_athlete['Gates (#)'].values[0])

    #rev_athlete

    st.metric(label="Athlete Name (Rev)", value=st.session_state['athlete_name_ref'])
    st.write("Rev Time (sec):", float(rev_athlete['total time (sec)'].values[0]))
    
    #compare_athlete
    st.metric(label="Athlete Name (Compare)", value=st.session_state['athlete_name_athlete_2'])
    st.write("Time (sec):", float(compare_athlete['total time (sec)'].values[0]))
    time_diff = round(float(compare_athlete['total time (sec)'].values[0]) - float(rev_athlete['total time (sec)'].values[0]), 2)

    st.write("Time Difference (sec):", time_diff)

    # Play video if venue is Adelboden and run is 2
    #if venue.lower() == 'adelboden' and run == 1:
    #    video_file = '/Users/marcgurber/SwissSki/SwissSki_Slalom/Ramon_Adelboden_7_1_2024.mov'
    #    video_bytes = open(video_file, 'rb').read()
    #    st.video(video_bytes)


    # Summary statistics for the 'total time (sec)' column
    return 1

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
    st.markdown(f"**Date:** {pd.to_datetime(venue_data_sorted['Date'].iloc[0]).strftime('%d.%m.%Y')}")
    st.session_state['athlete_name_ref'] = venue_data_sorted['athlete_name_ref'].iloc[0]
    st.session_state['athlete_name_athlete_2'] = venue_data_sorted['athlete_name_athlete_2'].iloc[0]
    
    summarized_data = summarize_data(athlete_data, selected_venue , run_number)
    #print(summarized_data)  

  # Display the dataset in Streamlit
    #st.dataframe(venue_data_sorted)

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

def merge_df(athlete_dataframe,course_slalom_dataframe):
    # Standardize date format
    athlete_dataframe['Date'] = pd.to_datetime(athlete_dataframe['Date'], format='%d,%m,%Y').dt.strftime('%Y-%m-%d')

    # Clean up gate column names by removing "(ms)"
    athlete_dataframe.columns = [col.replace(' (ms)', '') if 'Gate' in col else col for col in athlete_dataframe.columns]

    # Standardize venue names
    athlete_dataframe['Venue'] = athlete_dataframe['Venue'].str.replace(' Tahoe', '', regex=False)

    # Define metadata and gate columns
    metadata_columns = [
        'Date', 'Venue', 'Country', 'Course name', 'Course setter', 'Gates (#)', 
        'Turning Gates (#)', 'Start Altitude (m)', 'Finish Altitude (m)', 'Vertical Drop (m)', 
        'Start time', 'Weather', 'Snow', 'Temp Start (°C)', 'Temp Finish (°C)', 
        'Run', 'Nation', 'Best', 'Is Fastest?', 'ID', 'total time (sec)'
    ]
    gate_columns = [col for col in athlete_dataframe.columns if 'Gate' in col and 'total time' not in col]

    # Pivot databank to long format for each gate
    databank_long_df = athlete_dataframe.melt(
        id_vars=metadata_columns,
        value_vars=gate_columns,
        var_name='Gate',
        value_name='Time (ms)'
    )

    # Convert gate column to integer after removing 'Gate ' prefix
    databank_long_df['Gate'] = databank_long_df['Gate'].str.replace('Gate ', '').astype(int)

    # Separate into reference and other athlete times
    reference_df = databank_long_df[databank_long_df['Is Fastest?'] == 'yes'].copy()
    other_athlete_df = databank_long_df[databank_long_df['Is Fastest?'] == 'no'].copy()

    # Rename the Time column for clarity
    reference_df.rename(columns={'Time (ms)': 'ref_time'}, inplace=True)
    other_athlete_df.rename(columns={'Time (ms)': 'athlete_2_time'}, inplace=True)

    # Add athlete names for clarity
    reference_df['athlete_name_ref'] = reference_df['Best']
    other_athlete_df['athlete_name_athlete_2'] = other_athlete_df['Best']

    # Merge the datasets on common columns
    merged_df = pd.merge(
        reference_df[['Date', 'Venue', 'Run', 'Gate', 'ref_time', 'athlete_name_ref']],
        other_athlete_df[['Date', 'Venue', 'Run', 'Gate', 'athlete_2_time', 'athlete_name_athlete_2']],
        on=['Date', 'Venue', 'Run', 'Gate'],
        how='outer'
    )

    # Calculate absolute and relative differences
    merged_df['time_difference'] = merged_df['ref_time'] - merged_df['athlete_2_time']
    merged_df['relative_time_difference'] = (merged_df['time_difference'] / merged_df['ref_time']) * 100

    # Sort by Venue and Gate
    merged_df_sorted = merged_df.sort_values(by=['Venue', 'Gate']).reset_index(drop=True)

    # Save the sorted DataFrame to a CSV file
    #merged_df_sorted.to_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Sorted_Athlete_Times_by_Venue_and_Gate.csv', index=False, sep=';')  

    #print(merged_df_sorted.head())     # Display the first few rows of the merged DataFrame

    sorted_athlete_df = merged_df_sorted
    #course_slalom_dataframe = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Course_Slalom.csv', delimiter=';')

    # Standardize the date format in course_slalom_df to match sorted_athlete_df
    course_slalom_dataframe['Date'] = pd.to_datetime(course_slalom_dataframe['Date'], format='%d.%m.%y').dt.strftime('%Y-%m-%d')

    # Rename 'Gate (Nr)' to 'Gate' in course_slalom_df for consistency
    course_slalom_dataframe.rename(columns={'Gate (Nr)': 'Gate'}, inplace=True)

    # Merge sorted athlete times with course slalom data on 'Date', 'Venue', 'Run', and 'Gate'
    merged_course_athlete_df = pd.merge(
        sorted_athlete_df,
        course_slalom_dataframe,
        on=['Date', 'Venue', 'Run', 'Gate'],
        how='inner'
    )
    return merged_course_athlete_df

# Streamlit app
st.title("Slalom Course Relative Elevation Profile")
# Sidebar logic
st.session_state['analyse'] = "init"

with st.sidebar:
    uploaded_file1 = st.file_uploader("Upload - Databank_Slalom_23-24  -  (Athlete Time)", type="csv") 
    if uploaded_file1 is not None:
        athlete_data = pd.read_csv(uploaded_file1, delimiter=';')
        #st.write("filename:", uploaded_file1.name)
        st.write(athlete_data.head())
        
        uploaded_file2 = st.file_uploader("Upload - Course_Slalom - Course Data", type="csv")
        if uploaded_file2 is not None:
            slalom_data = pd.read_csv(uploaded_file2, delimiter=';')
            #st.write("filename:", uploaded_file2.name)
            st.write(slalom_data.head())

            # Dropdown for selecting venue
            venues = athlete_data['Venue'].unique()
            selected_venue = st.selectbox("Select a Venue", venues)

            # Dropdown for selecting run
            run_number = st.selectbox("Select Run Number", [1, 2])

            graph_data = merge_df(athlete_data, slalom_data)
        
            # Place the 'Analyse' button in the sidebar
            if st.button("Slalom Race Analyse"):
                st.session_state['analyse'] = "analyse"
                st.session_state['selected_venue'] = selected_venue
                st.session_state['run_number'] = run_number
                st.session_state['graph_data'] = graph_data  # Assuming the merge doesn't create very large data

            # Place the 'Analyse' button in the sidebar
            if st.button("Season Analysis"):
                st.session_state['analyse'] = "season_analysis"

if st.session_state['analyse'] == "init":
    st.title("Please upload the data files to begin the analysis.")


# Main window logic for plotting
if st.session_state['analyse'] == "analyse":
    fig = plot_relative_elevation_profile(st.session_state['graph_data'], st.session_state['selected_venue'], st.session_state['run_number'])
    st.plotly_chart(fig)
    # Reset the state if needed or allow for re-analysis
    if st.button("Clear Plot"):
        del st.session_state['analyse']  # This will remove the plot and reset the analysis state

if st.session_state['analyse'] == "season_analysis":
    del st.session_state['analyse']  # This will remove the plot and reset the analysis state
    st.write("Season Analysis")
    st.write("Test")
    # Add the season analysis