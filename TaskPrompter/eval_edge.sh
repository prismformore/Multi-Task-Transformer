#!/bin/bash

base=/data/hyeae/project/mtl/Multi-Task-Learning-PyTorch/eval_edge_files/
subdir=$1

mkdir "${base}${subdir}"
cp -r ../$1/results/edge /data/hyeae/project/mtl/Multi-Task-Learning-PyTorch/eval_edge_files/$1/edge
