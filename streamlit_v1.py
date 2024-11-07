import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

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
# Function to perform analysis and generate plots
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

    # Generate scatter plots with regression lines
    for feature, title in zip(selected_features, titles_selected):
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


# Initialization of the session state
if 'analyse' not in st.session_state:
    st.session_state['analyse'] = "init"

# App title
st.title("Slalom Course Relative Elevation Profile")

# Sidebar logic
with st.sidebar:
    uploaded_file1 = st.file_uploader("Upload - Databank_Slalom 23-24 - (Athlete Time)", type="csv")
    if uploaded_file1 is not None:
        athlete_data = pd.read_csv(uploaded_file1, delimiter=';')
        st.write(athlete_data.head())

        uploaded_file2 = st.file_uploader("Upload - Course_Slalom - Course Data", type="csv")
        if uploaded_file2 is not None:
            slalom_data = pd.read_csv(uploaded_file2, delimiter=';')
            st.write(slalom_data.head())

            if 'athlete_data' in locals() and 'slalom_data' in locals():
                venues = athlete_data['Venue'].unique()
                selected_venue = st.selectbox("Select a Venue", venues)
                run_number = st.selectbox("Select Run Number", [1, 2])

                # Assuming merge_df is defined to merge athlete_data and slalom_data appropriately
                graph_data = merge_df(athlete_data, slalom_data)  # Ensure merge_df function is correctly merging the data
                st.session_state['graph_data'] = graph_data  # Storing in session state

                if st.button("Slalom Race Analyse"):
                    st.session_state['analyse'] = "analyse"
                    st.session_state['selected_venue'] = selected_venue
                    st.session_state['run_number'] = run_number

                if st.button("Season Analysis"):
                    st.session_state['analyse'] = "season_analysis"

# Plotting logic
if st.session_state.get('analyse') == "analyse":
    st.write("Ready to plot:", st.session_state['graph_data'].head())  # Debugging output
    fig = plot_relative_elevation_profile(st.session_state['graph_data'], st.session_state['selected_venue'], st.session_state['run_number'])
    st.plotly_chart(fig)

# Use get() for safe access or check explicitly
if st.session_state.get('analyse', 'init') == "init":
    st.subheader("Please upload the data files to begin the analysis.")

if st.session_state.get('analyse') == "season_analysis":
    analyze_and_plot_features(st.session_state['graph_data'])
    st.write("Season Analysis Complete")
    st.session_state['analyse'] = "init"  # Resetting the state