#!/bin/bash
cd "$(dirname "$0")"

export SERVICE_NAME="loan_approval_prediction_classifier"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity | jq -r .Account)
export AWS_REGION=$(aws configure get region)


# Containerise the application

echo "Containerise bento and building it"

bentoctl build -b "${SERVICE_NAME}:latest" -f deployment_config.yaml
