import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
from scipy.stats import linregress
from scipy.stats import f_oneway
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.cm import ScalarMappable
from shapely.geometry import Point
import contextily as ctx
import geopandas as gpd
import cv2
from tempfile import NamedTemporaryFile

def merge_video_animation():  
    def side_by_side_blend(frame1, frame2, height, width):
        # Ensure both frames are valid and have the same height
        if frame1 is None:
            frame1 = np.zeros((height, width, 3), dtype=np.uint8)
        if frame2 is None:
            frame2 = np.zeros((height, width, 3), dtype=np.uint8)

        # Resize frames to have the same height
        frame1 = cv2.resize(frame1, (width, height))
        frame2 = cv2.resize(frame2, (width, height))

        # Concatenate frames side by side
        return np.concatenate((frame1, frame2), axis=1)

    # Open the video files
    global temp_video_run
    cap1 = cv2.VideoCapture(temp_video_run.name)
    cap2 = cv2.VideoCapture(temp_file_animation.name)

    # Ensure both videos are opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("One of the video files could not be opened")

    # Get properties from video1
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the desired fps
    fps = 60

    # Prepare output video writer with double width
    global temp_output_file
    temp_output_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

    # Process frames
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            break  # End of both videos

        # Apply the blend function
        output_frame = side_by_side_blend(frame1 if ret1 else None, frame2 if ret2 else None, height, width)
        out.write(output_frame)

    # Release resources
    cap1.release()
    cap2.release()
    out.release()
    return 


def run_animation(data,venue_name,date,run_number): 
    # Filter and prepare the data
    filtered_data = data[(data['Venue'] == 'Adelboden') & data['Latitude (°)'].notna() & data['Longitude (°)'].notna()].copy()
    filtered_data.loc[:, 'athlete_2_time'] = filtered_data['athlete_2_time'].fillna(0)

    # Calculate cumulative time difference
    filtered_data['cumulative_time'] = filtered_data['athlete_2_time'].cumsum()

    print(filtered_data)    

    # Calculate the total duration in seconds
    total_duration = filtered_data['cumulative_time'].iloc[-1]

    # Set desired frames per second
    desired_fps = 10  # or 30
    num_frames = int(total_duration * desired_fps)

    # Normalize relative_time_difference for color mapping
    norm = Normalize(vmin=filtered_data['relative_time_difference'].min(), vmax=filtered_data['relative_time_difference'].max())
    cmap = plt.get_cmap('RdYlGn')
    sm = ScalarMappable(cmap=cmap, norm=norm)
    colors = sm.to_rgba(filtered_data['relative_time_difference'])

    # Plotting setup
    fig, ax = plt.subplots()
    scat = ax.scatter(filtered_data['Longitude (°)'], filtered_data['Latitude (°)'], c=colors, s=100, edgecolor='k', alpha=0.8)
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    ax.set_title('Animation - Racename')

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)  # Hide ticks

    # Remove the border (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Red dot and time annotation
    blue_dot, = ax.plot([], [], 'bo', markersize=10)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    # Adjust the animation to match the cumulative time
    def update(frame):
        current_time = frame / desired_fps
        actual_frame = np.searchsorted(filtered_data['cumulative_time'], current_time, side='right') - 1
        blue_dot.set_data([filtered_data['Longitude (°)'].iloc[actual_frame]], [filtered_data['Latitude (°)'].iloc[actual_frame]])
        time_difference = filtered_data['time_difference'].iloc[actual_frame]
        time_info = f"Gate: {filtered_data['Gate'].iloc[actual_frame]}, Time Difference: {time_difference:.2f}s"
        time_text.set_text(time_info)
        time_text.set_position((0.98, 0.02))  # Set position to bottom right
        time_text.set_ha('right')  # Align text to the right
        # Set text color based on time_difference
        if time_difference < 0:
            time_text.set_color('red')
        else:
            time_text.set_color('green')
        return blue_dot, time_text

    # Animation setup
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=lambda: (blue_dot, time_text), blit=True)

    # Saving the animation temporarily for the Streamlit application
    global temp_file_animation
    temp_file_animation = NamedTemporaryFile(delete=False, suffix=".mp4")
    Writer = FFMpegWriter(fps=desired_fps, metadata=dict(artist='Me'), codec='libx264')
    ani.save(temp_file_animation.name, writer=Writer)
    temp_file_animation.close()

    return 1

def summarize_data(data, venue, run):
    rev_athlete = data[data['Best'] == st.session_state['athlete_name_ref']]
    compare_athlete = data[(data['Best'] == st.session_state['athlete_name_athlete_2'])]
   
     #Course data
    st.subheader("Race and Course Details:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Date", value=pd.to_datetime(rev_athlete['Date'].values[0]).strftime('%d.%m.%Y'))
        st.metric(label="Venue", value=rev_athlete['Venue'].values[0])
    with col2:
        st.metric(label="Course Name", value=rev_athlete['Course name'].values[0])
        st.metric(label="Course Setter", value=rev_athlete['Course setter'].values[0])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Gates", value=int(rev_athlete['Gates (#)'].values[0]))
    with col2:
        st.metric(label="Turning Gates", value=int(rev_athlete['Turning Gates (#)'].values[0]))
    with col3:
        st.metric(label="Temp Start (°C)", value=int(rev_athlete['Temp Start (°C)'].values[0]))
    
    #rev_athlete
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Athlete Name (Reference)", value=st.session_state['athlete_name_ref'])
    with col2:
        st.metric(label="Time (sec)", value=float(rev_athlete['total time (sec)'].values[0]))
    with col3:
        st.write("")

    #compare_athlete
    col1, col2, col3= st.columns(3)
    with col1:
        st.metric(label="Athlete Name (Compare)", value=st.session_state['athlete_name_athlete_2'])
    with col2:
        time_diff = round(float(compare_athlete['total time (sec)'].values[0]) - float(rev_athlete['total time (sec)'].values[0]), 2)
        st.metric(label="Time (sec)", value=float(compare_athlete['total time (sec)'].values[0]), delta=f"{time_diff} sec", delta_color="inverse")
    

    # Draw a thin white line
    st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)


    # Play video if venue is Adelboden and run is 2
    #if venue.lower() == 'adelboden' and run == 1:
    #    video_file = '/Users/marcgurber/SwissSki/SwissSki_Slalom/Ramon_Adelboden_7_1_2024.mov'
    #    video_bytes = open(video_file, 'rb').read()
    #    st.video(video_bytes)


    # Summary statistics for the 'total time (sec)' column
    return 1

def slalom_map(data):

    # Extract relevant columns
    latitude = data["Latitude (°)"]
    longitude = data["Longitude (°)"]
    altitude = data["Altitude (m)"]

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


    # Convert latitude and longitude to geospatial points and create GeoDataFrame
    geometry = [Point(xy) for xy in zip(longitude, latitude)]
    gdf = gpd.GeoDataFrame(venue_data_sorted, geometry=geometry)

    # Set the coordinate reference system to WGS84 (lat/lon) and transform to Web Mercator for map plotting
    gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

    # Calculate buffer for improved map boundaries
    buffer_factor = 0.2
    minx, miny, maxx, maxy = gdf.total_bounds
    x_buffer = (maxx - minx) * buffer_factor
    y_buffer = (maxy - miny) * buffer_factor

    # Plot the map view with a satellite image background
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf.plot(ax=ax, marker='o', color='green', linestyle='-', label='Slalom Course Path')
    ax.set_xlim(minx - x_buffer, maxx + x_buffer)
    ax.set_ylim(miny - y_buffer, maxy + y_buffer)
    
    # Setting the extent of the map to cover the buffered area before adding the base map
    ax.set_xlim(minx - x_buffer, maxx + x_buffer)
    ax.set_ylim(miny - y_buffer, maxy + y_buffer)
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.Esri.WorldImagery)

    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title('Map View of the Course')
    ax.legend()
    st.pyplot(fig)

     # Plot altitude profile with cumulative distance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_distances, altitude, marker='o', linestyle='-', color='blue')
    ax.set_xlabel('Cumulative Distance (m)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Profile by Cumulative Distance')
    ax.grid(True)
    st.pyplot(fig)

def adelboden_plot_course_map(data):
    # Sort by gate order
    data_sorted = data.sort_values(by="Gate")

    # Extract relevant columns
    latitude = data_sorted["Latitude (°)"]
    longitude = data_sorted["Longitude (°)"]
    altitude = data_sorted["Altitude (m)"]
    relative_time_diff = data_sorted["relative_time_difference"]

    # Convert latitude and longitude to geospatial points and create GeoDataFrame
    geometry = [Point(xy) for xy in zip(longitude, latitude)]
    gdf = gpd.GeoDataFrame(data_sorted, geometry=geometry)

    # Set the coordinate reference system to WGS84 (lat/lon) and transform to Web Mercator for map plotting
    gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

    # Calculate a buffer around the course to zoom out the view
    buffer_factor = 0.5  # Adjust this factor to increase or decrease the zoom level
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

    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title('Adelboden Slalom Course (Run 2) with Relative Time Difference Color Gradient')
    
    st.pyplot(fig)

def segment_analysis(graph_data):
    # Define the segments based on gate numbers
    num_gates = graph_data['Gate'].max()
    segment_size = num_gates // 3
    top_segment = range(1, segment_size + 1)
    middle_segment = range(segment_size + 1, 2 * segment_size + 1)
    bottom_segment = range(2 * segment_size + 1, num_gates + 1)

    # Categorize each gate into a segment
    def categorize_segment(gate):
        if gate in top_segment:
            return 'Top'
        elif gate in middle_segment:
            return 'Middle'
        elif gate in bottom_segment:
            return 'Bottom'
        return None

    graph_data['Segment'] = graph_data['Gate'].apply(categorize_segment)

    # Group and aggregate data
    detailed_segment_analysis = (
        graph_data.groupby(['Venue', 'Run', 'Gate', 'Segment'])['time_difference']
        .mean()
        .reset_index()
    )

    segment_summary = detailed_segment_analysis.groupby('Segment').agg(
        mean_time_difference=('time_difference', 'mean'),
        std_error=('time_difference', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        # Plotting with standard error
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.bar(segment_summary['Segment'], segment_summary['mean_time_difference'], yerr=segment_summary['std_error'],
               capsize=5, color='skyblue', alpha=0.8)
        ax.set_xlabel('Course Segment')
        ax.set_ylabel('Avg Time Diff (s)')
        ax.set_title('Avg Time Loss by Segment')
        st.pyplot(fig)

    with col2:
        # Boxplot of time differences by segment
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(x='Segment', y='time_difference', data=graph_data, palette='Set3', ax=ax)
        ax.set_xlabel('Course Segment')
        ax.set_ylabel('Time Diff (s)')
        ax.set_title('Time Differences by Segment')
        plt.tight_layout()
        st.pyplot(fig)

def analyze_and_plot_features(data):
    # Filter data for relevant columns and remove rows with missing values for the selected features
    features_data_selected = data[['time_difference', 'Turning Angle [°]', 'Offset [m]', 'Steepness [°]']].dropna()

    # Filter out data points where "Offset [m]" is greater than 150
    features_data_selected = features_data_selected[features_data_selected['Offset [m]'] < 150]

    # Exclude rows where time_difference is less than -1.0
    features_data_filtered = features_data_selected[features_data_selected['time_difference'] >= -1.0]

    # Perform linear regression and collect results for Turning Angle, Offset, and Steepness with filtered data
    selected_features = ['Turning Angle [°]', 'Offset [m]', 'Steepness [°]']
    filtered_results = {
        "Feature": [],
        "Slope": [],
        "Intercept": [],
        "R-squared": [],
        "p-value": [],
        "Significant (p < 0.05)": []
    }

    # Regression analysis for each feature against time_difference on the filtered dataset
    for feature in selected_features:
        slope, intercept, r_value, p_value, std_err = linregress(features_data_filtered['time_difference'], features_data_filtered[feature])
        # Append results to dictionary
        filtered_results["Feature"].append(feature)
        filtered_results["Slope"].append(slope)
        filtered_results["Intercept"].append(intercept)
        filtered_results["R-squared"].append(r_value**2)
        filtered_results["p-value"].append(p_value)
        filtered_results["Significant (p < 0.05)"].append(p_value < 0.05)

    # Convert results to a DataFrame
    filtered_results_df = pd.DataFrame(filtered_results)
    st.write("Filtered Significance of Selected Course Feature Relationships", filtered_results_df)

    # Plotting the relationships with the filtered data in separate figures with regression lines
    titles_selected = [
        "Impact of Turning Angle on Time Difference",
        "Impact of Offset on Time Difference",
        "Impact of Steepness on Time Difference"
    ]

    # Generate scatter plots with regression lines and display them side by side
    cols = st.columns(3)
    for col, feature, title in zip(cols, selected_features, titles_selected):
        with col:
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.regplot(x=features_data_filtered['time_difference'], y=features_data_filtered[feature], ax=ax, scatter_kws={'s': 10}, line_kws={'color': 'orange'})
            ax.set_xlabel("Time Difference (seconds)")
            ax.set_ylabel(feature)
            ax.set_title(f"{title}\nR-squared = {filtered_results_df[filtered_results_df['Feature'] == feature]['R-squared'].values[0]:.2f}")
            st.pyplot(fig)
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
        showlegend=False,
        width=1200,  # Increase the width
        height=800   # Increase the height
    )
    
    return fig

# App title
st.set_page_config(layout="wide")
st.title("Slalom Analysis")

# Initialization of the session state with default values
if 'file1_uploaded' not in st.session_state:
    st.session_state['file1_uploaded'] = False

if 'file2_uploaded' not in st.session_state:
    st.session_state['file2_uploaded'] = False

if 'analyse' not in st.session_state:
    st.session_state['analyse'] = "init"

# Sidebar logic
with st.sidebar:
    # File uploader for athlete data inside an expander
    with st.expander("Upload Athlete Times", expanded=not st.session_state.get('file1_uploaded', False)):
        uploaded_file1 = st.file_uploader("Upload - Databank Slalom 23-24 - (Athlete Time)", type="csv", key="uploader1")
        if uploaded_file1 is not None:
            athlete_data = pd.read_csv(uploaded_file1, delimiter=';')
            st.write(athlete_data)
            st.session_state['file1_uploaded'] = True

    # File uploader for course data inside an expander
    with st.expander("Upload Course Data", expanded=not st.session_state.get('file2_uploaded', False)):
        uploaded_file2 = st.file_uploader("Upload - Course Slalom - Course Data", type="csv", key="uploader2")
        if uploaded_file2 is not None:
            slalom_data = pd.read_csv(uploaded_file2, delimiter=';')
            st.write(slalom_data)
            st.session_state['file2_uploaded'] = True

    # Additional logic when both files are uploaded
    if st.session_state['file1_uploaded'] and st.session_state['file2_uploaded']:
        # Text input for user to specify what they want to analyze
        st.subheader("Please Select your Analysis")
        # Button to initiate slalom race analysis
        
        if st.button("Run Video Animation"):
            st.session_state['analyse'] = "animation"

        if st.button("Analyse Slalom Race"):
            st.session_state['analyse'] = "analyse"
            
        # Button to initiate slalom map analysis
        if st.button("Analyse Slalom Course"):
            st.session_state['analyse'] = "map"

        # Button to initiate season analysis
        if st.button("Analyse Season Data"):
                st.session_state['analyse'] = "season_analysis"    

        # Button to trigger snow effect
        if st.button("Let it Snow!"):
            st.snow()

# Plotting logic
if st.session_state.get('analyse') == "analyse":
    venues = athlete_data['Venue'].unique()
    selected_venue = st.selectbox("Select a Venue", venues)
    run_number = st.selectbox("Select Run Number", [1, 2])
    st.session_state['selected_venue'] = selected_venue
    st.session_state['run_number'] = run_number
    graph_data = merge_df(athlete_data, slalom_data)
    st.session_state['graph_data'] = graph_data
    fig = plot_relative_elevation_profile(st.session_state['graph_data'], st.session_state['selected_venue'], st.session_state['run_number'])
    st.plotly_chart(fig)
    if selected_venue.lower() == 'adelboden' and run_number == 2:
        venue_data_sorted = graph_data[(graph_data['Venue'] == selected_venue)].sort_values(by="Gate")
        adelboden_plot_course_map(venue_data_sorted)
        # Video file uploader
        with st.expander("Upload Race Video", expanded=True):
            uploaded_video = st.file_uploader("Upload - Race Video", type=["mp4", "mov", "avi"], key="uploader_video")
            if uploaded_video is not None:
                #video merge
                 # Button to initiate season analysis
                st.video(uploaded_video)
                delay = st.slider("Select Video Delay (seconds)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
                if st.button("Analyse merge Video"):
                    st.session_state['analyse'] = "merge_Video"   
                    video_file = '/Users/marcgurber/SwissSki/SwissSki_Slalom/MEILLARD Loic_Adelboden_animation..mp4'
                    video_bytes = open(video_file, 'rb').read()
                    st.video(video_bytes)
                    # Add a download button for the video
                    st.download_button(
                        label="Download Race Video",
                        data=video_bytes,
                        file_name="race_video.mp4",
                        mime="video/mp4"
                    )
                    
if st.session_state.get('analyse') == "animation":
    venue_data = slalom_data.dropna(subset=["Longitude (°)", "Latitude (°)"])  # Ensure no missing values in Longitude or Latitude
    venue_data_sorted = venue_data.sort_values(by="Gate (Nr)")
    venue_data_sorted = venue_data_sorted.sort_values(by= "Venue")
    venues = venue_data_sorted['Venue'].unique()
    selected_venue = st.selectbox("Select a Venue", venues)
    venue_data_sorted = venue_data[(venue_data['Venue'] == selected_venue)].sort_values(by="Gate (Nr)")
    date = venue_data_sorted['Date'].unique()
    selected_date = st.selectbox("Select a date", date)
    run_number = st.selectbox("Select Run Number", [1, 2])
    st.session_state['run_number'] = run_number
    venue_data_sorted = venue_data[(venue_data['Date'] == selected_date)].sort_values(by="Gate (Nr)")
    st.session_state['slalom_data'] = venue_data_sorted
    st.session_state['selected_venue'] = selected_venue
    st.session_state['selected_date'] = selected_date
    graph_data = merge_df(athlete_data, slalom_data)
    st.session_state['graph_data'] = graph_data
    run_animation(st.session_state['graph_data'], st.session_state['selected_venue'], st.session_state['selected_date'], st.session_state['run_number'])
    
    global temp_file_animation
    with open(temp_file_animation.name, 'rb') as video_file:
        st.video(video_file.read())
    
    with st.expander("Upload Race Video", expanded=True):
        uploaded_video = st.file_uploader("Upload - Race Video", type=["mp4", "mov", "avi"], key="uploader_video")
        if uploaded_video is not None:
            # Display the uploaded video
            st.video(uploaded_video)
            global temp_video_run
            with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_run:
                temp_video_run.write(uploaded_video.read())
                temp_video_path = temp_video_run.name
                
            delay = st.slider("Select Video Delay (seconds)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            if st.button("Analyse merge Video"):
                st.session_state['analyse'] = "merge_Video"
                # Save the uploaded video to a temporary file
                merge_video_animation()
                with open(temp_output_file.name, 'rb') as video_file:
                    st.video(video_file.read())

                # Add a download button for the video
                with open(temp_output_file.name, 'rb') as video_file:
                    st.download_button(
                        label="Download Merged Video",
                        data=video_file.read(),
                        file_name="merged_video.mp4",
                        mime="video/mp4"
                    )

# Use get() for safe access or check explicitly
if st.session_state.get('analyse', 'init') == "init":
    st.subheader("Please upload the data files to begin the analysis.")

if st.session_state.get('analyse') == "map":
    #drop data without GPS coordinates
    venue_data = slalom_data.dropna(subset=["Longitude (°)", "Latitude (°)"])  # Ensure no missing values in Longitude or Latitude
    venue_data_sorted = venue_data.sort_values(by="Gate (Nr)")
    venue_data_sorted = venue_data_sorted.sort_values(by= "Venue")
    venues = venue_data_sorted['Venue'].unique()
    selected_venue = st.selectbox("Select a Venue", venues)
    venue_data_sorted = venue_data[(venue_data['Venue'] == selected_venue)].sort_values(by="Gate (Nr)")
    date = venue_data_sorted['Date'].unique()
    selected_date = st.selectbox("Select a date", date)
    venue_data_sorted = venue_data[(venue_data['Date'] == selected_date)].sort_values(by="Gate (Nr)")
    st.session_state['slalom_data'] = venue_data_sorted
    st.session_state['selected_venue'] = selected_venue
    st.session_state['selected_date'] = selected_date
    slalom_map(venue_data_sorted)

if st.session_state.get('analyse') == "season_analysis":
    graph_data = merge_df(athlete_data, slalom_data)
    st.session_state['graph_data'] = graph_data
    analyze_and_plot_features(st.session_state['graph_data'])
    segment_analysis(st.session_state['graph_data'])
    st.write("Season Analysis Complete")
    #st.session_state['analyse'] = "init"  # Resetting the state
