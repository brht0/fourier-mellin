import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('../build/release/src'))
import cv2_fourier_mellin

# input_video = "../images/video.mp4"
input_video = "../images/recording.mp4"
output_video_path = "stabilized.mp4"
cap = cv2.VideoCapture(input_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
scaledown_factor = 4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width//scaledown_factor, height//scaledown_factor), isColor=False)

fm = cv2_fourier_mellin.FourierMellinWithReference(width//scaledown_factor, height//scaledown_factor)

referenceIsSet = False
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width//scaledown_factor, height//scaledown_factor))
    if not referenceIsSet:
        fm.set_reference(frame)
        referenceIsSet = True
    else:
        stable_frame, transform = fm.register_image(frame)
        out.write(stable_frame.astype(np.uint8))

cap.release()
out.release()
cv2.destroyAllWindows()
