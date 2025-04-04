import cv2
import os

# Configuration
input_video = r'C:\Users\anish\OneDrive\Desktop\motion estimation\frames\input.mp4'
output_folder = './extracted_frames'
max_frames = 50

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise Exception(f"Error opening video file: {input_video}")

frame_count = 0

while frame_count < max_frames:
    # Read next frame
    ret, frame = cap.read()
    
    # Break loop if we can't retrieve any more frames
    if not ret:
        break
    
    # Convert from BGR to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Save frame as JPEG
    output_path = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(output_path, frame_rgb)
    
    frame_count += 1

# Clean up
cap.release()
print(f"Successfully saved {frame_count} frames to {output_folder}")