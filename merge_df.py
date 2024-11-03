import pandas as pd

# Load the databank file with the correct delimiter
databank_slalom_df = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Databank_Slalom_23-24.csv', delimiter=';')

# Standardize date format
databank_slalom_df['Date'] = pd.to_datetime(databank_slalom_df['Date'], format='%d,%m,%Y').dt.strftime('%Y-%m-%d')

# Clean up gate column names by removing "(ms)"
databank_slalom_df.columns = [col.replace(' (ms)', '') if 'Gate' in col else col for col in databank_slalom_df.columns]

# Standardize venue names
databank_slalom_df['Venue'] = databank_slalom_df['Venue'].str.replace(' Tahoe', '', regex=False)

# Define metadata and gate columns
metadata_columns = [
    'Date', 'Venue', 'Country', 'Course name', 'Course setter', 'Gates (#)', 
    'Turning Gates (#)', 'Start Altitude (m)', 'Finish Altitude (m)', 'Vertical Drop (m)', 
    'Start time', 'Weather', 'Snow', 'Temp Start (°C)', 'Temp Finish (°C)', 
    'Run', 'Nation', 'Best', 'Is Fastest?', 'ID', 'total time (sec)'
]
gate_columns = [col for col in databank_slalom_df.columns if 'Gate' in col and 'total time' not in col]

# Pivot databank to long format for each gate
databank_long_df = databank_slalom_df.melt(
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
merged_df_sorted.to_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Sorted_Athlete_Times_by_Venue_and_Gate.csv', index=False, sep=';')  

print(merged_df_sorted.head())     # Display the first few rows of the merged DataFrame


sorted_athlete_df = merged_df_sorted
course_slalom_df = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Course_Slalom.csv', delimiter=';')

# Standardize the date format in course_slalom_df to match sorted_athlete_df
course_slalom_df['Date'] = pd.to_datetime(course_slalom_df['Date'], format='%d.%m.%y').dt.strftime('%Y-%m-%d')

# Rename 'Gate (Nr)' to 'Gate' in course_slalom_df for consistency
course_slalom_df.rename(columns={'Gate (Nr)': 'Gate'}, inplace=True)

# Merge sorted athlete times with course slalom data on 'Date', 'Venue', 'Run', and 'Gate'
merged_course_athlete_df = pd.merge(
    sorted_athlete_df,
    course_slalom_df,
    on=['Date', 'Venue', 'Run', 'Gate'],
    how='inner'
)

# Save the merged DataFrame to a CSV file for download
merged_course_athlete_df.to_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_Course_and_Athlete_Times_by_Gate.csv', index=False, sep=';') 