import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Load the data
data = pd.read_csv('/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_Course_and_Athlete_Times_by_Gate.csv', delimiter=';')

# Define the segments based on gate numbers
num_gates = data['Gate'].max()
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

data['Segment'] = data['Gate'].apply(categorize_segment)

# Group by Venue, Run, Gate, and Segment to create finer-grained data points
detailed_segment_analysis = (
    data.groupby(['Venue', 'Run', 'Gate', 'Segment'])['time_difference']
    .mean()
    .reset_index()
)

# Calculate mean and standard error for each segment
segment_summary = detailed_segment_analysis.groupby('Segment').agg(
    mean_time_difference=('time_difference', 'mean'),
    std_error=('time_difference', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
).reset_index()

# Plotting with standard error
plt.figure(figsize=(8, 6))
plt.bar(segment_summary['Segment'], segment_summary['mean_time_difference'], yerr=segment_summary['std_error'], 
        capsize=5, color='skyblue', alpha=0.8)
plt.xlabel('Course Segment')
plt.ylabel('Average Time Difference (seconds)')
plt.title('Average Time Loss by Course Segment with Standard Error')
plt.show()

# Boxplot of time differences by segment
plt.figure(figsize=(10, 8))
sns.boxplot(x='Segment', y='time_difference', data=data, palette='Set3')
plt.xlabel('Course Segment')
plt.ylabel('Time Difference (seconds)')
plt.title('Boxplot of Time Differences by Course Segment')
plt.show()