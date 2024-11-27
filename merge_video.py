import cv2
import numpy as np

def side_by_side_blend(frame1, frame2, height, width):
    # Ensure both frames are valid and have the same height
    if frame1 is None:
        frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    if frame2 is None:
        frame2 = np.zeros((height, width, 3), dtype=np.uint8)

    # Resize frames to have the same height
    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    # Concatenate frames side by side
    return np.concatenate((frame1, frame2), axis=1)

# Open the video files
cap1 = cv2.VideoCapture('/Users/marcgurber/SwissSki/SwissSki_Slalom/MEILLARD Loic_Adelboden.mp4')
cap2 = cv2.VideoCapture('/Users/marcgurber/SwissSki/01_Swiss_Ski/Adelboden_Animation_blue.mp4')

# Ensure both videos are opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    raise ValueError("One of the video files could not be opened")

# Get properties from video1
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the desired fps
fps = 60

# Prepare output video writer with double width
out = cv2.VideoWriter('output_side_by_side.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

# Process frames
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 and not ret2:
        break  # End of both videos

    # Apply the blend function
    output_frame = side_by_side_blend(frame1 if ret1 else None, frame2 if ret2 else None, height, width)
    out.write(output_frame)

# Release resources
cap1.release()
cap2.release()
out.release()
print("Done!")
