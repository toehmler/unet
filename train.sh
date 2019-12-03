#!/bin/sh
python3 train.py unet_4 0 25 25 32 0.25
python3 train.py unet_4 25 55 30 32 0.25
python3 train.py unet_4 55 90 35 32 0.25
python3 train.py unet_4 90 130 40 32 0.25
python3 train.py unet_4 130 180 50 32 0.25


