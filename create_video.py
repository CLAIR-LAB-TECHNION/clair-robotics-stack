import cv2
import os
import re

# Config
image_folder = 'rgb_frames'
output_video_path = 'output_video_with_s3e_planning.mp4'
frame_rate = 2
start_frame = 214  # Include from frame_50.png onwards

# Regex to match filenames like "frame_42.png"
pattern = re.compile(r'^frame_(\d+)\.png$')

# Collect valid images with their frame numbers
images = []
for fname in os.listdir(image_folder):
    match = pattern.match(fname)
    if match:
        frame_num = int(match.group(1))
        if frame_num >= start_frame:
            images.append((frame_num, fname))

# Sort by frame number
images.sort(key=lambda x: x[0])

# Make sure there are images to process
if not images:
    raise ValueError(f"No images found starting from frame_{start_frame}.png")

# Get frame size from the first image
first_image_path = os.path.join(image_folder, images[0][1])
frame = cv2.imread(first_image_path)
height, width, _ = frame.shape

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

# Write frames
for _, filename in images:
    path = os.path.join(image_folder, filename)
    frame = cv2.imread(path)
    video.write(frame)

video.release()
print(f"Video created from frame_{start_frame}.png onwards: {output_video_path}")