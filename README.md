# Fourier-Mellin Image Registration Python Library using OpenCV

This repository implements the Fourier-Mellin transform for image registration and video stabilization using semilog polar coordinates with OpenCV. The implementation is written in C++, but python bindings are provided. For information about the pipeline, see [this article](https://ieeexplore.ieee.org/document/506761). Tested on Ubuntu 20.04.

Note that the image registration only works effectively for Eucledian/similar transformations without affine or perspective distortions.

## Video Stabilization




https://github.com/brht0/fourier-mellin/assets/90235713/c5524042-6bc6-46b5-837e-adb62ddcf9b1

Original video can be found [on Youtube](https://www.youtube.com/watch?v=mQxnB2X26CI) (No affiliation)



## Image Registration

![lena_transform_demo](https://github.com/brht0/fourier-mellin/assets/90235713/9db0cc26-581f-40db-b34c-3516e759f960)

Image registration with a transformed image, overlayed.

## Requirements (Ubuntu 20.04)

- OpenCV (tested with OpenCV 4.10.0)
- Python3 (tested with python3.8)
- CMake (version 3.14 or higher)
- C++20 compliant compiler (e.g. g++-11)

You can install the build requirements with the following command:

```
sudo apt install cmake build-essential g++-11
```

### Extra dependencies

- pybind11

## Adding to a Python Project

Disclaimer: This repository is an unstable build, and it is highly discouraged to do a system-wide install. Please use a virtual enviroment.

### Cloning the repository

```
# Python project's root directory
mkdir ext
git clone https://github.com/brht0/fourier-mellin ext/cv2_fourier_mellin
```

### Building the package inside a virtual enviroment

```
python3 -m venv venv
source venv/bin/activate
pip install ext/cv2_fourier_mellin
```

### Image Matching Demo

```python
import cv2
import numpy as np
import cv2_fourier_mellin

reference = cv2.imread('lenna.png')
transformed = cv2.imread('lenna_transformed.png')

reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
transformed_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

reference_gray = np.float32(reference_gray)
transformed_gray = np.float32(transformed_gray)

rows, cols = reference.shape[:2]

fm = cv2_fourier_mellin.FourierMellin(cols, rows)

transformed_reference, transform = fm.register_image(reference, transformed)

overlay = cv2.addWeighted(transformed, 0.5, transformed_reference, 0.5, 0.0, dtype=cv2.CV_32F)
cv2.imwrite("overlay.jpg", overlay)
cv2.imwrite("transformed.jpg", transformed)
```

### Video Stabilization Demo

```python
import cv2
import numpy as np
import cv2_fourier_mellin

input_video = "raw_video.mp4"
output_video = "stabilized_video.mp4"

cap = cv2.VideoCapture(input_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

fm = cv2_fourier_mellin.FourierMellinContinuous(width, height, 0.1)

print(f"Starting video stabilization for {input_video} into {output_video}.")
frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    print(f"Stabilizing frame {frame_counter:03d}.")

    stable_frame, transform = fm.register_image(frame)
    if np.prod(stable_frame.shape) > 0:
        out.write(stable_frame.astype(np.uint8))

cap.release()
out.release()
print(f"Saved sabilized video to {output_video}.")
```

## Building without pip

This is not required for usage inside Python. You may skip this step.

```
mkdir -p build/release
cd build/release
cmake ../.. -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_MODULE=ON
cd ../..
cmake --build build/release -j 4
```

## Todo

- Major cleanup
- Unit tests
- Install/Uninstall
- Proper Readme
- Optimization
- CLI
- CUDA with OpenCV
- Documentation
- Option to calculate transform with lower resolution
