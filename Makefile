# Run bento training and create service and model locally
create_bento_artifacts:
	pipenv run bash ./create_bento_artifacts.sh

# Push bento image to aws
push_image:
	pipenv run bash ./deployment/push_bento_image.sh

# Plan terraform prod
plan_prod:
	bash ./deployment/run_plan.sh

# Apply terraform prod local machine
apply_prod:
	bash ./deploy/run_apply.sh

# Destroy terraform prod local machine
destroy_prod:
	bash ./deploy/run_destroy.sh

# For local setup
setup:
	pip install -U pip
	pip install pipenv
	pipenv install