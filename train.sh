#!/bin/sh
python3 train.py unet_7 0 40 30 128 0.25 
python3 train.py unet_7 40 80 30 128 0.25 
python3 train.py unet_7 80 120 30 128 0.25 
python3 train.py unet_7 120 160 30 128 0.25 

