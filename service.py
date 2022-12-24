import os

import bentoml
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel


MODEL_NAME = os.getenv("MODEL_NAME", "loan_approval_prediction_model")
SERVICE_NAME = os.getenv("SERVICE_NAME", "loan_approval_prediction_classifier")
class LoanApprovalServiceData(BaseModel):

    gender : str = "Male"
    married : str = "Yes"
    dependents: str = "1"
    education: str = "Graduate"
    self_employed: str =  "No"
    applicantincome: int = 4583
    coapplicantincome: float = 1508.0
    loanamount: float = 128.0
    loan_amount_term: float = 360.0
    credit_history: float = 1.0
    property_area: str = "Rural"


model_ref = bentoml.sklearn.get(
    tag_like=f"{MODEL_NAME}:latest"
)

preprocessor = model_ref.custom_objects["preprocessor"]
imputator = model_ref.custom_objects["imputator"]
transformer = model_ref.custom_objects["transformer"]

runner = model_ref.to_runner()

svc = bentoml.Service(
    name=f"{SERVICE_NAME}",
    runners=[runner]
)

@svc.api(input=JSON(pydantic_model=LoanApprovalServiceData), output=JSON())
async def classify(raw_request):
    """Function to classify and make loan prediction"""

    app_data = pd.DataFrame(raw_request.dict(), index=[0])
    vector_processed = preprocessor.transform(app_data)

    vector_imputed = imputator.transform(vector_processed)
    vector_imputed_df = pd.DataFrame(data=vector_imputed, columns=imputator.get_feature_names_out())
    vector_transformed_df = transformer.transform(vector_imputed_df)

    prediction = await runner.predict_proba.async_run(vector_transformed_df)
    result = round(prediction[0][1],3)

    if result > 0.5:
        return {
            "status": 200,
            "probability_of_getting_loan": result,
            "loan_approved": "YES"
        }
    else:
       return {
            "status": 200,
            "probability_of_getting_loan": result,
            "loan_approved": "NO"
        }
