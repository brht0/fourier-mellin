# Fourier-Mellin

Todo

## Requirements

- OpenCV
- PyBind11
- g++-11 (or other C++20 compliant compiler)

```
pip install pybind11 g++-11
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

Todo