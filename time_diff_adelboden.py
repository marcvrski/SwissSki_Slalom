import pandas as pd
import numpy as np

# Load your data (replace 'file_path' with your actual file path)
data_course = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Course_Slalom.csv', delimiter=';')
data_athletes = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Databank_Slalom_23-24.csv', delimiter=';')


# Filter data for Adelboden only and drop rows with missing Longitude data
adelboden_data_athletes = data_athletes[(data_athletes['Venue'] == 'Adelboden') & (data_athletes['Run'] != 1)]

adelboden_data_course = data_course[(data_course['Venue'] == 'Adelboden') & (data_course['Run'] == 1)]

# Pivot adelboden_data_athletes
pivot_athletes = adelboden_data_athletes.pivot(index='Bib', columns='Gate', values='Time')

# Merge with adelboden_data_course by gates
merged_data = adelboden_data_course.merge(pivot_athletes, left_on='Gate', right_index=True, how='left')

print(adelboden_data_course)
print(adelboden_data_athletes)
print(merged_data)  

