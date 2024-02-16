#!/bin/sh

echo "Training the Greedy InfoMax Model on vision data (stl-10)"
python -m vision.main_vision --grayscale --download_dataset --save_dir vision_experiment

echo "Testing the Greedy InfoMax Model for image classification"
python -m vision.downstream_classification --grayscale --model_path ./logs/vision_experiment --model_num 299
