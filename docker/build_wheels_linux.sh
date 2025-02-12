#!/bin/sh

if [ "${PWD##*/}" != "fourier-mellin" ]; then
    echo "Please run inside the root of the project fourier-mellin"
    exit
fi

export CIBW_MANYLINUX_X86_64_IMAGE="ghcr.io/htoik/fourier-mellin/manylinux_x86_64:latest"
export CIBW_MANYLINUX_PYPY_X86_64_IMAGE="ghcr.io/htoik/fourier-mellin/manylinux_x86_64:latest"
export CIBW_MANYLINUX_AARCH64_IMAGE="ghcr.io/htoik/fourier-mellin/manylinux_aarch64:latest"
export CIBW_MANYLINUX_PYPY_AARCH64_IMAGE="ghcr.io/htoik/fourier-mellin/manylinux_aarch64:latest"
# export CIBW_MANYLINUX_X86_64_IMAGE=""
# export CIBW_MANYLINUX_PYPY_X86_64_IMAGE=""
# export CIBW_MANYLINUX_AARCH64_IMAGE=""
# export CIBW_MANYLINUX_PYPY_AARCH64_IMAGE=""
export CIBW_MANYLINUX_I686_IMAGE=""
export CIBW_MANYLINUX_PYPY_I686_IMAGE=""
export CIBW_MANYLINUX_PPC64LE_IMAGE=""
export CIBW_MANYLINUX_S390X_IMAGE=""
export CIBW_MANYLINUX_ARMV7L_IMAGE=""
export CIBW_MUSLLINUX_X86_64_IMAGE=""
export CIBW_MUSLLINUX_AARCH64_IMAGE=""
export CIBW_MUSLLINUX_I686_IMAGE=""
export CIBW_MUSLLINUX_PPC64LE_IMAGE=""
export CIBW_MUSLLINUX_S390X_IMAGE=""
export CIBW_MUSLLINUX_ARMV7L_IMAGE=""

export CIBW_PLATFORM="linux"
export CIBW_ARCHS="aarch64 x86_64"
export CIBW_SKIP="cp36-* cp36-* *_i686 *-musllinux_* pp*-manylinux_aarch64"

python -m cibuildwheel --output-dir wheelhouse .
