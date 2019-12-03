#!/bin/sh
python3 train.py unet_5 0 25 25 32 0.25
python3 train.py unet_5 25 55 30 32 0.25
python3 train.py unet_5 55 90 35 32 0.25
python3 train.py unet_5 90 130 40 32 0.25
python3 train.py unet_5 130 180 50 32 0.25


