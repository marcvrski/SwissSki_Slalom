import cv2
import numpy as np

def multiply_blend(frame1, frame2):
    # Ensure frame2 is resized to match frame1's dimensions
    frame2_resized = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    # Convert frames to float and normalize
    float1 = frame1.astype(float) / 255.0
    float2 = frame2_resized.astype(float) / 255.0
    # Multiply the frames and scale back to 255
    blended_frame = (float1 * float2 * 255).astype(np.uint8)
    return blended_frame

# Open the video files
cap1 = cv2.VideoCapture('/Users/marcgurber/SwissSki/SwissSki_Slalom/MEILLARD Loic_Adelboden.mp4')
cap2 = cv2.VideoCapture('/Users/marcgurber/SwissSki/SwissSki_Slalom/Adelboden_Animation.mp4')

# Get properties from video1
fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

# Prepare output video writer
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps1, (width, height))

# Process frames
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1:
        break  # End of video1

    if not ret2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video2
        ret2, frame2 = cap2.read()

    # Apply the blend function
    output_frame = multiply_blend(frame1, frame2)
    out.write(output_frame)

# Release resources
cap1.release()
cap2.release()
out.release()
