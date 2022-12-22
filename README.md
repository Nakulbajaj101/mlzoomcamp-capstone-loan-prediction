# Loan Prediction Service
The prupose of this project is to provide a loan prediction for people so that banks and 
finance providers can make unbiased decisions, and make sure prople get the loan when they deserve it.
The service like this may make getting a loan process unbiased from human intervention.

# Motivation 
In developing countries there is not enough data available about people, and usually banks make 
biased decisions on their own perception on who should or should not get the loan. For example in 
some countries women are perceived to be house wives and may be perceived as not being able to afford
to a house. A service like this will look beyond the cultural barriers


# Data Source 
The dataset is available on Analytics Vidya and has been downloaded and made available

# Build With
The section covers tools used to run the project
1. Python for data exploration with pandas, seaborn and matplotlib
2. Python for ML pipeline with sklearn, xgboost and bentoml
3. Bentoml framework in python to build the deployment service
4. Bash for orchatrating model training, building deployment and pushing to ECR on the cloud
5. AWS Serverless and api gateway
6. Locust for local load testing of bentoml api
7. Streamlit for prediction service app

# Project structure
![alt text](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/ProjectTree.png)

# How to run training

1. In the root directory run `pipenv shell`
2. Then run `python training.py`

# How to build the service and containerize it

1. Run the following commands in order in bash from the root directory
Note: Make sure `jq` is installed and `docker` is installed and running

Activate the pipenv virtual env shell
```bash
pipenv shell
```


```bash
echo "Building the bento"
bentoml build

# Containerise the application

echo "Containerise bento and building it"

export MODEL_TAG=$(bentoml get loan_approval_prediction_classifier:latest -o json | jq -r .version)
cd ~/bentoml/bentos/$SERVICE_NAME && bentoml containerize "${SERVICE_NAME}":latest
```

# How to run the project end to end on AWS Fargate

Note: Make sure `jq` is installed and `docker` is installed and running, also make sure AWS profile is configured locally which has privelages to create ECR repo and create an image

Activate the pipenv virtual env shell
```bash
pipenv shell
```

Run the following bash script in the root directory
```bash
bash ./create_bento_artifacts.sh
```

This will train the model, build the service, containerize it and take it to ecr repo
The name of the repo will be `loan_approval_prediction_classifier:latest`

Now we can follow the [video 9.7](https://www.youtube.com/watch?v=7gI1UH31xb4&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=97) to deploy it behind lambda 


# Data exploration, model selection and EDA

1. Open the `notebooks/data_exploration.ipynb` file and `training.ipynb` file to see data exploration and model selection strategy
2. Open the `training.ipynb` and `training.py` script to see model selection and model building as ML pipelines

Note: Four models were trained, Decision tree, logistic regression and XGBoost, and best one was chosen with highest ROC.