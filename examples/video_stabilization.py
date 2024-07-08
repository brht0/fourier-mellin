import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('../build/debug/src'))
import cv2_fourier_mellin

input_video = "../images/recording.mp4"
output_video_path = "stabilized.mp4"
cap = cv2.VideoCapture(input_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
scaledown_factor = 4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width//scaledown_factor, height//scaledown_factor), isColor=True)

fm = cv2_fourier_mellin.FourierMellinContinuous(width//scaledown_factor, height//scaledown_factor)

transformSum = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width//scaledown_factor, height//scaledown_factor))

    stable_frame, transform = fm.register_image(frame)
    if np.prod(stable_frame.shape) > 0:
        print(transformSum)
        stable_frame2 = cv2_fourier_mellin.get_transformed(frame, transformSum).astype(np.uint8)
        out.write(stable_frame2)

        transformSum = transform + transformSum
    else:
        transformSum = transform

cap.release()
out.release()
cv2.destroyAllWindows()
