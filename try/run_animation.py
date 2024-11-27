import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Load your data
data_path = '/Users/marcgurber/SwissSki/01_Swiss_Ski/SwissSki_Slalom/try/Merged_Course_and_Athlete_Times_by_Gate.csv'
data = pd.read_csv(data_path, delimiter=';')

# Filter and prepare the data
filtered_data = data[(data['Venue'] == 'Adelboden') & data['Latitude (°)'].notna() & data['Longitude (°)'].notna()].copy()
filtered_data.loc[:, 'athlete_2_time'] = filtered_data['athlete_2_time'].fillna(0)

# Calculate cumulative time difference
filtered_data['cumulative_time'] = filtered_data['athlete_2_time'].cumsum()

print(filtered_data)    

# Calculate the total duration in seconds
total_duration = filtered_data['cumulative_time'].iloc[-1]

# Set desired frames per second
desired_fps = 60  # or 30
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

# Saving the animation
Writer = FFMpegWriter(fps=desired_fps, metadata=dict(artist='Me'), codec='libx264')
ani.save('run_animation' + "racename" +'.mp4', writer=Writer)

# Display the animation
plt.legend()
plt.show()

