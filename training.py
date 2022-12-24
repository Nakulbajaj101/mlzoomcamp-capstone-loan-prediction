import logging
import os
from typing import List

import bentoml
import pandas as pd
import xgboost as xgb
from feature_engine.creation import MathFeatures, RelativeFeatures
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation.categorical import CategoricalImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing.preprocessors import (BinaryEncoder, CleanStrings,
                                         CustomColumnMapping, CustomRowMapping,
                                         DivideColumnValue,
                                         GreaterThanBinaryEncoding,
                                         MultiplyColumnValue,
                                         PrefixStringEncoding)

logging.basicConfig(level=logging.INFO)
MODEL_NAME = os.getenv("MODEL_NAME", "loan_approval_prediction_model")


# Importing the data
def read_data(filepath=""): 
    """Function to read data"""
    
    data = pd.read_csv(filepath_or_buffer=filepath)
    return data



# Cleaning the columns
def make_columns_consistent(df: pd.DataFrame) -> pd.DataFrame:
    """Function to make column names consistent"""

    data = df.copy()
    columns = [cols.lower().replace(" ","_") for cols in data]
    data.columns = columns

    return data

# Define preprocessor pipeline

def get_preprocessor_pipeline(clean_strings_cols: List, column_mappings: dict, thousand_multiplier_cols: List, month_divider_cols: List,
                              month_encoder_cols: List, month_encoder_prefix: str, data_mapping_cols: List, data_mappings: dict,
                              binary_encoder_cols: List, binary_encoder_value: str, gender_encoder_cols: List, gender_encoder_value: str) -> Pipeline:
    """Function to define and return preprocessor pipeline"""

    preprocessor_pipeline = Pipeline(
        steps=[
            ('clean_strings', 
            CleanStrings(column_list=clean_strings_cols)
            ),
            ('column_mappings',
            CustomColumnMapping(column_mapping=column_mappings)
            ),
            ('thousand_multiplier', 
            MultiplyColumnValue(column_list=thousand_multiplier_cols, 
                                multiply_by=1000)
            ),
            ('month_divider',
            DivideColumnValue(column_list=month_divider_cols, 
                            divide_by=12)
            ),
            ('prefix_month_encoder',
            PrefixStringEncoding(column_list=month_encoder_cols,
                                string_val=month_encoder_prefix)
            ),
            ('data_mappings', 
            CustomRowMapping(column_list=data_mapping_cols, 
                            column_value_mapping=data_mappings)
            ),
            ('binary_common_encoder', 
            BinaryEncoder(column_list=binary_encoder_cols, 
                        value=binary_encoder_value)
            ),
            ('gender_encoder', 
            BinaryEncoder(column_list=gender_encoder_cols, 
                        value=gender_encoder_value)
            )
        ])

    return preprocessor_pipeline


def get_imputation_pipeline(cat_cols: List) -> Pipeline:
    """Function to define transformer pipeline"""

    imputation_pipeline = Pipeline(
        steps=[
            (
                'cat_imputer',
                CategoricalImputer(fill_value='missing',
                                   variables=cat_cols)
            ),   
            (
                'rare_label_encoding',
                RareLabelEncoder(tol=0.01,
                                 n_categories=8,
                                 variables=cat_cols)
            ),
            (   
                'ohe_encoding',
                OneHotEncoder(variables=cat_cols)
            ),
            (   
                'iterative_imputer',
                IterativeImputer(estimator=ExtraTreesRegressor(bootstrap=True,
                                                               n_jobs=-1,
                                                               random_state=1),
                                 initial_strategy="median",
                                 random_state=1,
                                 max_iter=20)
            )]
    )

    return imputation_pipeline

def feature_creation_pipeline(math_features: List, reference_column: str, greater_than_val: float, column_suffix: str) -> Pipeline:
    
    feature_creation_pipeline = Pipeline(
        steps=[
            ( 
                'create_total_income_feature',
                MathFeatures(variables=math_features,
                             func=["sum"])

            ),
            (
                'create_income_loan_ratio',
                RelativeFeatures(variables=[f"sum_{math_features[0]}_{math_features[1]}"],
                                 reference=[reference_column],
                                 func=["div"])
            ),
            (
                'income_loan_ratio_greater_than_flag',
                GreaterThanBinaryEncoding(column_list=[f"sum_{math_features[0]}_{math_features[1]}_div_{reference_column}"],
                                          greater_than=greater_than_val,
                                          column_suffix=column_suffix)
            )
        ]
    )

    return feature_creation_pipeline


def model_training(X_train: pd.DataFrame, y_train: pd.Series):
    """Function to train and optimise the four models"""

    # Decision Tree
    d_param_grid = {
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8,10,20],
        'min_samples_leaf' : [1,3,5,10,20],
        'criterion' : ['gini', 'entropy'],
        'random_state' : [1], 
        'class_weight' : ['balanced', None]
    }
    d_clf = DecisionTreeClassifier()

    # Random Forest
    rf_param_grid = { 
        'n_estimators': [100],
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8,10,20],
        'min_samples_leaf' : [1,3,5,10,20],
        'criterion' : ['gini', 'entropy'],
        'random_state' : [1], 
        'class_weight' : [None, 'balanced']
    }

    rf_clf = RandomForestClassifier(n_jobs=-1)

    # Xgboost
    xgb_params = {
        'eta': [0.05, 0.1, 0.2],
        'max_depth': [4,5,6,7,8,10,20],
        'min_child_weight': [1,3,5,10,20],
        'n_estimators': [5, 10, 20, 50, 100],
        'objective':['binary:logistic'],
        'seed': [1],
        'verbosity': [1]
    }

    xgb_clf = xgb.XGBClassifier()

    d_clf_cv = GridSearchCV(estimator=d_clf, param_grid=d_param_grid, cv=5, scoring='roc_auc')
    d_clf_cv.fit(X_train, y_train)

    logging.info("Decision tree optimised")

    rf_clf_cv = GridSearchCV(estimator=rf_clf, param_grid=rf_param_grid, cv=5, scoring='roc_auc')
    rf_clf_cv.fit(X_train, y_train)

    logging.info("Random Forest optimised")

    xgb_clf_cv = GridSearchCV(estimator=xgb_clf, param_grid=xgb_params, cv=5, scoring='roc_auc')
    xgb_clf_cv.fit(X_train, y_train)

    logging.info("xgboost classifier optimised")

    d_best_params = d_clf_cv.best_params_
    rf_best_params = rf_clf_cv.best_params_
    xgb_best_params = xgb_clf_cv.best_params_

    logging.info("Training the best models")
    d_best_clf = DecisionTreeClassifier(**d_best_params)
    rf_best_clf = RandomForestClassifier(**rf_best_params)
    xgb_best_clf = xgb.XGBClassifier(**xgb_best_params)

    d_best_clf.fit(X_train, y_train)
    rf_best_clf.fit(X_train, y_train)
    xgb_best_clf.fit(X_train, y_train)

    return d_best_clf, rf_best_clf, xgb_best_clf



def select_best_model(d_tree_clf: DecisionTreeClassifier,
                      xgb_clf: xgb.XGBClassifier,
                      rf_clf: RandomForestClassifier,
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> tuple:

    """Function to evaluate models and return best model"""

    def evaluate_roc(model, X_val, y_val):
        """Evaluation function to return recall"""

        predictions = model.predict_proba(X_val)[:,1]
        roc_auc = roc_auc_score(y_val, predictions)
        return roc_auc


    X_test = X_test.copy()
    y_test = y_test.copy()

    d_roc_auc = evaluate_roc(d_tree_clf, X_val=X_test, y_val=y_test)
    rf_roc_auc = evaluate_roc(rf_clf, X_val=X_test, y_val=y_test)
    xgb_roc_auc = evaluate_roc(xgb_clf, X_val=X_test, y_val=y_test)


    model_performances = {
        "decision_tree" : {
            "model" : d_tree_clf,
            "roc_auc" : d_roc_auc
        },
        "xgboost" : {
            "model" : xgb_clf,
            "roc_auc" : xgb_roc_auc
        },
        "random_forest" : {
            "model" : rf_clf,
            "roc_auc" : rf_roc_auc
        }
    }

    logging.info(f"Models and their best performance scores \n {model_performances}")

    best_model = sorted(model_performances.items(), reverse=True, key=lambda score: score[1]['roc_auc'])[0]

    return best_model


def create_bento(best_model: tuple, preprocessor: Pipeline, imputator: Pipeline, transformer: Pipeline):
    """Function to create bento"""

    if best_model[0] == 'xgboost':
        logging.info(f"Should use bentoml xgboost framework")
        
        model = best_model[1]['model']
        bentoml.xgboost.save_model(
        name=f'{MODEL_NAME}',
        model=model,
        custom_objects={
            "preprocessor": preprocessor,
            "imputator": imputator,
            "transformer": transformer
        },
        signatures={
            "predict_proba":{
                "batchable": True,
                "batch_dim": 0
            }
        }
        )

    else:
        logging.info(f"Should use bentoml scikit learn framework")
        model = best_model[1]['model']
        bentoml.sklearn.save_model(
        name=f'{MODEL_NAME}',
        model=model,
        custom_objects={
            "preprocessor": preprocessor,
            "imputator": imputator,
            "transformer": transformer
        },
        signatures={
            "predict_proba":{
                "batchable": True,
                "batch_dim": 0
            }
        },
        )
    
    logging.info(f"Bento created")

if __name__ == "__main__":

    logging.info("Reading the data")
    data = read_data(filepath="dataset/loan_prediction.csv")

    logging.info("Making columns consistent")
    data = make_columns_consistent(df=data)

    id_col = 'loan_id'
    target = 'loan_status'

    data_columns = [cols for cols in data if cols not in [id_col, target]]
    X = data[data_columns].copy()
    y = data[target].apply(lambda val: val.lower()).map({"y":1, "n":0})

    logging.info("Splitting the data into train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    clean_strings_cols = [cols for cols in data if data[cols].dtype == 'object' and cols not in [id_col, target]]

    data_mapping_cols = ['graduate', 'property_area']


    thousand_multiplier_cols = ['loan_amount']

    month_divider_cols = ['loan_amount_term']

    month_encoder_cols = ['loan_amount_term']

    month_encoder_prefix = 'month'

    binary_encoder_cols = ['married', 'graduate', 'self_employed']

    binary_encoder_value='yes'

    gender_encoder_cols = ['gender']

    gender_encoder_value = 'male'

    column_mappings = {'applicantincome':'applicant_income', 
                    "coapplicantincome":"co_application_income", 
                    "loanamount":"loan_amount",
                    "education":"graduate"}

    data_mappings = { 'graduate': {'graduate': 'yes', 'not_graduate_graduate': 'no'},
    'property_area': { 'rural': 'rural',
                        'semiurban': 'semi_urban',
                        'urban': 'urban'}}

    math_features = ["applicant_income", "co_application_income"]
    reference_column = "loan_amount"
    greater_than_val = 0.027
    column_suffix = "027"


    

    logging.info("Building the preprocessor")
    preprocessor_pipeline=get_preprocessor_pipeline(clean_strings_cols=clean_strings_cols, 
                                                    column_mappings=column_mappings,
                                                    thousand_multiplier_cols=thousand_multiplier_cols,
                                                    month_divider_cols=month_divider_cols,
                                                    month_encoder_cols=month_encoder_cols, 
                                                    month_encoder_prefix=month_encoder_prefix,
                                                    data_mapping_cols=data_mapping_cols, 
                                                    data_mappings=data_mappings,
                                                    binary_encoder_cols=binary_encoder_cols,
                                                    binary_encoder_value=binary_encoder_value,
                                                    gender_encoder_cols=gender_encoder_cols,
                                                    gender_encoder_value=gender_encoder_value)

    logging.info("Running the preprocessor")
    X_train_preprocessed_data = preprocessor_pipeline.fit_transform(X_train)
    X_test_preprocessed_data = preprocessor_pipeline.transform(X_test)

    feature_cols = [cols for cols in X_train_preprocessed_data]
    cat_cols = [cols for cols in feature_cols if X_train_preprocessed_data[cols].dtype == 'object']
    
    logging.info("Defining data imputation and encoding pipelines")
    imputation_pipeline = get_imputation_pipeline(cat_cols=cat_cols)
    
    logging.info("Running the encoder and imputation")
    imputed_data_train = imputation_pipeline.fit_transform(X_train_preprocessed_data)
    imputed_data_test = imputation_pipeline.transform(X_test_preprocessed_data)

    X_train_imputed = pd.DataFrame(data=imputed_data_train, columns=imputation_pipeline.get_feature_names_out())
    X_test_imputed = pd.DataFrame(data=imputed_data_test, columns=imputation_pipeline.get_feature_names_out())

    logging.info("Running feature creation pipeline")
    feature_pipeline = feature_creation_pipeline(
        math_features=math_features,
        reference_column=reference_column,
        greater_than_val=greater_than_val,
        column_suffix=column_suffix
    )
    X_train_transformed = feature_pipeline.fit_transform(X_train_imputed)
    X_test_transformed = feature_pipeline.transform(X_test_imputed)

    logging.info("Training the three models and hypertuning")
    d_tree_clf, rf_clf, xgb_clf = model_training(X_train=X_train_transformed, y_train=y_train)

    logging.info("Selecting the best model")
    best_model = select_best_model(d_tree_clf=d_tree_clf,
                                   xgb_clf=xgb_clf,
                                   rf_clf=rf_clf,
                                   X_test=X_test_transformed,
                                   y_test=y_test)


    logging.info("Preparing the bento")
    create_bento(best_model=best_model, 
                 preprocessor=preprocessor_pipeline,
                 imputator=imputation_pipeline,
                 transformer=feature_pipeline
                 )
