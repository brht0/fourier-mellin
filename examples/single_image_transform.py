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

high_pass, apodization_window = brht_fourier_mellin.get_filters(cols, rows)

log_polar_map = brht_fourier_mellin.create_log_polar_map(cols, rows)

log_polar_img_referenece = brht_fourier_mellin.process_image(reference_gray, high_pass, apodization_window, log_polar_map)
log_polar_img_transformed = brht_fourier_mellin.process_image(transformed_gray, high_pass, apodization_window, log_polar_map)

transform = brht_fourier_mellin.register_image(reference_gray, transformed_gray, log_polar_img_referenece, log_polar_img_transformed, log_polar_map)

transformed_reference = brht_fourier_mellin.get_transformed(reference, transform)

overlay = cv2.addWeighted(transformed, 0.5, transformed_reference, 0.5, 0.0, dtype=cv2.CV_32F)
cv2.imwrite("overlay.jpg", overlay)
