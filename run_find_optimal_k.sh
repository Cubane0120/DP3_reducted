#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


taskset -c 0-6 python 3D-Diffusion-Policy/find_optimal_k.py

# sudo systemctl isolate multi-user.target
# sudo systemctl isolate graphical.target
