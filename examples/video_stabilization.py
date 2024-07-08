import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('../build/release/src'))
import cv2_fourier_mellin

input_video = "../images/shaky.mp4"
output_video_path = "stabilized_continuous.mp4"

cap = cv2.VideoCapture(input_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

fm = cv2_fourier_mellin.FourierMellinContinuous(width, height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    stable_frame, transform = fm.register_image(frame)
    if np.prod(stable_frame.shape) > 0:
        out.write(stable_frame.astype(np.uint8))

cap.release()
out.release()
