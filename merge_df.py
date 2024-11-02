import pandas as pd

# Load the uploaded files with semicolon as the delimiter
course_slalom_df = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Course_Slalom.csv', delimiter=';')
databank_slalom_df = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Databank_Slalom_23-24.csv', delimiter=';')

# Standardize the date format in both dataframes to YYYY-MM-DD for consistency
course_slalom_df['Date'] = pd.to_datetime(course_slalom_df['Date'], format='%d.%m.%y').dt.strftime('%Y-%m-%d')
databank_slalom_df['Date'] = pd.to_datetime(databank_slalom_df['Date'], format='%d,%m,%Y').dt.strftime('%Y-%m-%d')

# Rename gate columns to a consistent format by removing " (ms)" suffix
databank_slalom_df.columns = [col.replace(' (ms)', '') if 'Gate' in col else col for col in databank_slalom_df.columns]

# Standardize venue names by removing extra descriptors like "Tahoe"
databank_slalom_df['Venue'] = databank_slalom_df['Venue'].str.replace(' Tahoe', '', regex=False)

# Identify metadata and gate columns
metadata_columns = [
    'Date', 'Venue', 'Country', 'Course name', 'Course setter', 'Gates (#)', 
    'Turning Gates (#)', 'Start Altitude (m)', 'Finish Altitude (m)', 'Vertical Drop (m)', 
    'Start time', 'Weather', 'Snow', 'Temp Start (°C)', 'Temp Finish (°C)', 
    'Run', 'Nation', 'Best', 'Is Fastest?', 'ID', 'total time (sec)'
]
gate_columns = [col for col in databank_slalom_df.columns if 'Gate' in col and 'total time' not in col]

# Pivot databank_slalom_df to have a long format for Gate timings
databank_long_df = databank_slalom_df.melt(
    id_vars=metadata_columns,
    value_vars=gate_columns,
    var_name='Gate',
    value_name='Time (ms)'
)

# Convert gate column to integer after removing 'Gate ' prefix
databank_long_df['Gate'] = databank_long_df['Gate'].str.replace('Gate ', '').astype(int)

# Rename 'Gate (Nr)' to 'Gate' in course_slalom_df for consistency
course_slalom_df.rename(columns={'Gate (Nr)': 'Gate'}, inplace=True)
databank_long_df.to_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/athlete_time_test.csv', index=False, sep=';')  # Save as CSV with semicolon delimiter

print(databank_long_df)  # Display the first few rows of the long-formatted dataframe


exit()  # Exit the script to avoid running the next part

# Perform an outer join to ensure no data is dropped
merged_df = pd.merge(
    course_slalom_df, databank_long_df,
    on=['Date', 'Venue', 'Run', 'Gate'],
    how='outer'
)

# Save the merged dataframe to a CSV file for download
merged_df.to_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_DF.csv', index=False, sep=';')  # Save as CSV with semicolon delimiter
print(merged_df.head())  # Display the first few rows of the merged dataframe

#--------------

import pandas as pd

# Load the data with semicolon as the delimiter
adelboden_df = pd.read_csv('/mnt/data/Adelboden_DF.csv', delimiter=';')

# Separate the reference athlete (Is Fastest = yes) and the other athlete (Is Fastest = no)
reference_df = adelboden_df[adelboden_df['Is Fastest?'] == 'yes'].copy()
other_athlete_df = adelboden_df[adelboden_df['Is Fastest?'] == 'no'].copy()

# Rename the Time columns for clarity before merging
reference_df.rename(columns={'Time (ms)': 'ref_time'}, inplace=True)
other_athlete_df.rename(columns={'Time (ms)': 'athlete_2_time'}, inplace=True)

# Perform a simplified merge using only essential columns to match records for each gate
merged_df = pd.merge(
    reference_df[['Date', 'Discipline', 'Venue', 'Race (R) or Training (TR)', 'Gate', 'Run', 'ref_time']],
    other_athlete_df[['Date', 'Discipline', 'Venue', 'Race (R) or Training (TR)', 'Gate', 'Run', 'athlete_2_time']],
    on=['Date', 'Discipline', 'Venue', 'Race (R) or Training (TR)', 'Gate', 'Run'],
    how='outer'
)

# Calculate the absolute and relative time differences
merged_df['time_difference'] = merged_df['ref_time'] - merged_df['athlete_2_time']
merged_df['relative_time_difference'] = (merged_df['time_difference'] / merged_df['ref_time']) * 100

# Save the updated DataFrame to a CSV file for download
merged_df.to_csv('/mnt/data/Merged_Athlete_Times_with_Differences.csv', index=False)
