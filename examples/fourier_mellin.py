import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('../build/debug/src'))
import brht_fourier_mellin

reference = cv2.imread('../images/reference.jpg')
transformed = cv2.imread('../images/transformed.jpg')

reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
transformed_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

reference_gray = np.float32(reference_gray)
transformed_gray = np.float32(transformed_gray)

rows, cols = reference.shape[:2]

fm = brht_fourier_mellin.FourierMellin(cols, rows)

transformed, transform = fm.register_image(reference_gray, transformed_gray)
print(transform)

overlay = cv2.addWeighted(transformed, 0.5, reference_gray, 0.5, 0.0, dtype=cv2.CV_32F)
cv2.imwrite("overlay.jpg", overlay)
cv2.imwrite("transformed.jpg", transformed)
