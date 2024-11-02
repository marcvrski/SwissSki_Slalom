import pandas as pd

adelboden_data = pd.DataFrame()

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

# Perform an outer join to ensure no data is dropped
merged_df = pd.merge(
    course_slalom_df, databank_long_df,
    on=['Date', 'Venue', 'Run', 'Gate'],
    how='outer'
)

# Save the merged dataframe to a CSV file for download with semicolon as the delimiter
merged_df.to_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_DF.csv', index=False, sep=';')

# Filter data for Adelboden only and drop rows with missing Longitude data
adelboden_data = merged_df[(merged_df['Venue'] == 'Adelboden')]

print(adelboden_data) 

# Save the merged dataframe to a CSV file for download with semicolon as the delimiter
adelboden_data.to_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Adelboden_DF.csv', index=False, sep=';')


