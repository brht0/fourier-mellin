import os
import sys

package_dir = os.path.dirname(__file__)
so_file = os.path.join(package_dir, 'cv2_fourier_mellin.so')

if not os.path.exists(so_file):
    raise ImportError("The shared object file cv2_fourier_mellin.so does not exist in the package directory.")

from .cv2_fourier_mellin import *