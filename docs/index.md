---
layout: page
homepage: true
hide: true
title: libmolgrid
---

# Overview

> libmolgrid is a library to generate tensors from molecular data, with properties that make its output particularly suited to machine learning. 

libmolgrid abstracts basic input generation functionality used in our related project, [gnina](https://github.com/gnina/gnina), with applications reported in several papers including [Protein-Ligand Scoring with Convolutional Neural Networks](https://arxiv.org/abs/1612.02751).

It's implemented in C++/CUDA with Python bindings, with first-class integration for PyTorch, Caffe, and Keras.

# Installation

  Recommended:

  ```bash
  pip install molgrid
  ```

  Install via conda is also supported:

  ```bash
  conda install -c jsunseri molgrid
  ```

  Alternatively, you can build from source:

  ```bash
  pip3 install numpy pytest pyquaternion
  ```

[Install cmake 3.12 or higher.](https://cmake.org/install/)

[Install CUDA.](https://developer.nvidia.com/cuda-downloads)

Install OpenBabel 3.0 (Not yet released, build from [master](https://github.com/openbabel/openbabel)).

  ```bash
  git clone https://github.com/gnina/libmolgrid.git
  cd libmolgrid
  mkdir build
  cd build
  cmake ..
  make -j8
  sudo make install
  ```
