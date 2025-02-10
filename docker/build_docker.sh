#!/bin/sh

# manylinux-x86_64
docker build -f Dockerfile.manylinux_x86_64 -t ghcr.io/htoik/fourier-mellin/manylinux_x86_64:latest .
# manylinux-aarch64
docker buildx build --platform linux/arm64 -f Dockerfile.manylinux_aarch64 -t ghcr.io/htoik/fourier-mellin/manylinux_aarch64:latest .
