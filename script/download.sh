#!/bin/bash
orientaion_path="https://zenodo.org/record/5163061/files/deepTMpred-b.pth"
deeptmpred_path="https://zenodo.org/record/5163061/files/orientaion-b.pth"
mkdir model_files
wget -P model_files "${orientaion_path}"
wget -P model_files "${deeptmpred_path}"
echo "all model files are downloaded successfully!"
