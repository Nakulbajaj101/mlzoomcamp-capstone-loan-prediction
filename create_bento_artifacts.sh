#!bin/bash

################SET AWS DEFAULT PROFILE#################
# export AWS_PROFILE=your_profile_with_ECS_PERMISSIONS #
# Watch the video https://www.youtube.com/watch?v=aF-TfJXQX-w&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=72 #

####Once the profile is ready you can take the image to AWS####

export MODEL_NAME="loan_approval_prediction_model"
export SERVICE_NAME="loan_approval_prediction_classifier"

# Running the training
echo "Running training"
python training.py

# Building the application

echo "Building the bento"
bentoml build
