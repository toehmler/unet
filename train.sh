#!/bin/sh
python3 train.py unet_3 0 40 50 32 0.25
python3 train.py unet_3 40 80 50 32 0.25
python3 train.py unet_3 80 120 50 32 0.25
python3 train.py unet_3 120 160 50 32 0.25
python3 train.py unet_3 160 190 50 32 0.25

