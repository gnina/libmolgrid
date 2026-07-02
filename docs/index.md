---
layout: page
homepage: true
hide: true
title: libmolgrid
---

# Overview

> libmolgrid is a library to generate tensors from molecular data, with properties that make its output particularly suited to machine learning. 

libmolgrid abstracts basic input generation functionality used in our related project, [gnina](https://github.com/gnina/gnina), with applications reported in several papers including [Protein-Ligand Scoring with Convolutional Neural Networks](https://arxiv.org/abs/1612.02751).

It's implemented in C++ with Python bindings. This fork keeps the upstream CUDA backend available and adds a Metal backend for Apple Silicon builds.

# Installation

## PIP

```bash
pip install molgrid
```

## conda

```bash
conda install -c gnina molgrid
```

## Build from Source

```bash
apt install git build-essential libboost-all-dev python3-pip rapidjson-dev
pip3 install numpy pytest pyquaternion
```

[Install cmake 3.12 or higher.](https://cmake.org/install/)

[Install CUDA.](https://developer.nvidia.com/cuda-downloads)

Install OpenBabel 3.0 (build from [master](https://github.com/openbabel/openbabel) if needed).

`apt install libeigen3-dev libboost-all-dev`

```bash
git clone https://github.com/gnina/libmolgrid.git
cd libmolgrid
mkdir build
cd build
cmake ..
make -j8
sudo make install
```

## macOS / Apple Silicon

For this fork, macOS builds can use the Metal backend when CUDA is not available.

```bash
brew install cmake boost open-babel rapidjson eigen python
pip3 install numpy pytest pyquaternion
```

Xcode Command Line Tools are required for the Metal toolchain:

```bash
xcode-select --install
```

Build with Metal explicitly enabled:

```bash
git clone https://github.com/fnachon/libmolgrid.git
cd libmolgrid
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DLIBMOLGRID_ENABLE_CUDA=OFF
make -j$(sysctl -n hw.logicalcpu)
sudo make install
```

To build with the upstream CUDA backend instead, disable Metal and use a CUDA-enabled toolchain:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DLIBMOLGRID_ENABLE_METAL=OFF
```
