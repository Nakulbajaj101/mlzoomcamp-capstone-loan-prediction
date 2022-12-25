#!/bin/bash

cd "$(dirname "$0")"

# run terraform for ${ENV}
echo "Initialising terraform"
terraform init -backend-config="key=mlzoomcamp-capstone-one.tfstate" --reconfigure

echo "Running terraform plan"
terraform plan -var-file="bentoctl.tfvars"
