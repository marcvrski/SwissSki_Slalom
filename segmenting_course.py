import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/mnt/data/Merged_Course_and_Athlete_Times_by_Gate.csv', delimiter=';')

# Determine the number of gates to divide into segments
num_gates = data['Gate'].max()
segment_size = num_gates // 3

# Define the segments based on gate numbers
top_segment = range(1, segment_size + 1)
middle_segment = range(segment_size + 1, 2 * segment_size + 1)
bottom_segment = range(2 * segment_size + 1, num_gates + 1)

# Add a new column to categorize each gate into one of the segments
def categorize_segment(gate):
    if gate in top_segment:
        return 'Top'
    elif gate in middle_segment:
        return 'Middle'
    elif gate in bottom_segment:
        return 'Bottom'
    return None

data['Segment'] = data['Gate'].apply(categorize_segment)

# Group by segment and calculate average time difference for each segment
segment_analysis = data.groupby('Segment')['time_difference'].mean().reset_index()

# Plotting the average time difference per segment
plt.figure(figsize=(8, 6))
plt.bar(segment_analysis['Segment'], segment_analysis['time_difference'], color='skyblue')
plt.xlabel('Course Segment')
plt.ylabel('Average Time Difference (seconds)')
plt.title('Average Time Loss by Course Segment')
plt.show()
