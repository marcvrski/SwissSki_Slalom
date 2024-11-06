import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Load your data
data_path = '/Users/marcgurber/SwissSki/SwissSki_Slalom/Merged_Course_and_Athlete_Times_by_Gate.csv'
data = pd.read_csv(data_path, delimiter=';')

# Filter rows with missing geolocation data and where venue is 'Adelboden'
filtered_data = data[(data['Venue'] == 'Adelboden') & data['Latitude (°)'].notna() & data['Longitude (°)'].notna()].copy()
filtered_data.loc[:, 'athlete_2_time'] = filtered_data['athlete_2_time'].fillna(0)

# Total duration for the animation in seconds
total_duration = filtered_data['athlete_2_time'].sum()

# Calculate FPS based on total duration and number of frames
fps = len(filtered_data) / total_duration

# Create the figure and axis
fig, ax = plt.subplots()
ax.plot(filtered_data['Longitude (°)'], filtered_data['Latitude (°)'], label="Course Path", color='blue')
ax.scatter(filtered_data['Longitude (°)'], filtered_data['Latitude (°)'], color='blue')  # show gates
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Adelboden Slalom Course Animation')

# Red dot to represent the athlete
red_dot, = ax.plot([], [], 'ro', markersize=10)

# Time annotation
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def update(frame):
    red_dot.set_data([filtered_data['Longitude (°)'].iloc[frame]], [filtered_data['Latitude (°)'].iloc[frame]])
    time_info = f"Gate: {filtered_data['Gate'].iloc[frame]}, Time: {filtered_data['athlete_2_time'].iloc[frame]}s"
    time_text.set_text(time_info)
    return red_dot, time_text

def frames():
    for i in range(len(filtered_data)):
        yield i
        time.sleep(filtered_data['athlete_2_time'].iloc[i])

# Create animation
ani = FuncAnimation(fig, update, frames=frames, init_func=lambda: (red_dot, time_text), blit=True, save_count=len(filtered_data))

# Set up formatting for the movie files with FFmpeg
Writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), codec='libx264')

# Save the animation as MP4
ani.save('Adelboden_Animation.mp4', writer=Writer)

# Show the animation
plt.legend()
plt.show()
