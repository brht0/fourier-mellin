# Fourier-Mellin Image Registration Python Library using OpenCV

This repository implements the Fourier-Mellin transform for image registration and video stabilization using semilog polar coordinates with OpenCV. The implementation is written in C++, but python bindings are provided. For information about the pipeline, see [this article](https://ieeexplore.ieee.org/document/506761). Tested on Ubuntu 20.04.

## Requirements (Ubuntu 20.04)

- OpenCV (tested with OpenCV 4.10.0)
- Python3 (tested with python3.8)
- CMake (version 3.14 or higher)
- C++20 compliant compiler (e.g. g++-11)

<!-- You can install the requirements with the following command:
```
sudo apt install cmake build-essential g++-11
``` -->

## Extra dependencies

- pybind11

## Building

```
mkdir -p build/release
cd build/release
cmake ../.. -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_BUILD_TYPE=Release
cd -
cmake --build build/release -j 4
```

## Usage

See the examples in the `examples/` folder. You can run the examples from the root directory:

```
python3 ./examples/video_stabilization.py
```

## Todo

- Major cleanup
- Unit tests
- Install/Uninstall
- Proper Readme
- Optimization
- CLI
