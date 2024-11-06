from moviepy.editor import VideoFileClip, CompositeVideoClip

# Load the videos
video2 = VideoFileClip("/Users/marcgurber/SwissSki/SwissSki_Slalom/Adelboden_Animation.mp4").resize(height=480)  # Example resizing
video1 = VideoFileClip("/Users/marcgurber/SwissSki/SwissSki_Slalom/MEILLARD Loic_Adelboden.mov").resize(width=240)  # Different resize approach

# Delay the start of the second video
video2 = video2.set_start(5)  # Starts playing after 5 seconds

# Set position of each video in the output
video2 = video2.set_position((50, 100))  # Position video1
video2 = video2.set_position((400, 200))  # Position video2 after the delay

# Create a composite video, specifying the duration of the output to be long enough to include all clips
final_clip = CompositeVideoClip([video2, video1.set_duration(video2.duration)], size=(1280, 720))

# Write the result to a file
final_clip.write_videofile("custom_layout_video.mp4", codec="libx264", fps=24)