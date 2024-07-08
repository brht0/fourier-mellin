# Fourier-Mellin Image Registration Python Library using OpenCV

This repository implements the Fourier-Mellin transform for image registration and video stabilization using semilog polar coordinates with OpenCV. The implementation is written in C++, but python bindings are provided. For information about the pipeline, see [this article](https://ieeexplore.ieee.org/document/506761).

## Requirements

- OpenCV
- pybind11
- C++20 compliant compiler (e.g. g++-11)

You can install the requirements with the following command:

```
sudo apt install pybind11 g++-11
```

## Building

```
mkdir -p build/release
cd build/release
cmake ../.. -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_BUILD_TYPE=Release
cd -
cmake --build build/release -j 4
```

## Usage

See the `examples/` folder.

## Todo

- Major cleanup
- Unit tests
- Install/Uninstall
- Proper Readme
- Optimization
