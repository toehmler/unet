#!/bin/sh
sudo apt update 
sudo apt install python3-pip
curl "https://d1vvhvl2y92vvt.cloudfront.net/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
pip3 install SimpleITK keras tensrflow scikit-learn tqdm scikit-image
sudo apt install mosh



