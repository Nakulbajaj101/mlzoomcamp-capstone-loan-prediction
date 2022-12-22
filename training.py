import logging
import os
from typing import List

import bentoml
import numpy as np
import pandas as pd
import xgboost as xgb
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation.categorical import CategoricalImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing.preprocessors import (BinaryEncoder, CleanStrings,
                                         CustomColumnMapping, CustomRowMapping,
                                         DivideColumnValue,
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


def get_transformer_pipeline(cat_cols: List) -> Pipeline:
    """Function to define transformer pipeline"""

    transformer_pipeline = Pipeline(
        steps=[
            ('cat_imputer',
            CategoricalImputer(fill_value='frequent',
                                variables=cat_cols)
            ),   
            ('rare_label_encoding',
            RareLabelEncoder(tol=0.01,
                            n_categories=8,
                            variables=cat_cols)
            ),
            ('ohe_encoding',
            OneHotEncoder(variables=cat_cols)
            ),
            ('scaling_data',
                MinMaxScaler()
            ),
            ('knn_imputer',
            KNNImputer(add_indicator=True)
            )
        ]
    )

    return transformer_pipeline


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


    # Logistic Regression
    lr_param_grid = {
        "C":np.logspace(-3,3,7), 
        "max_iter": [500, 1000,2000, 5000],
        'class_weight' : ['balanced', None],
        'random_state' : [1]
        } 
        
    lr_clf = LogisticRegression()


    # Random Forest
    rf_param_grid = { 
        'n_estimators': [100,200,300],
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8,10,20],
        'min_samples_leaf' : [1,3,5,10,20],
        'criterion' : ['gini', 'entropy'],
        'random_state' : [1], 
        'class_weight' : ['balanced', None]
    }

    rf_clf = RandomForestClassifier(n_jobs=-1)

    # Xgboost
    xgb_params = {
        'eta': [0.05, 0.1, 0.2],
        'max_depth': [4,5,6,7,8,10,20],
        'min_child_weight': [1,3,5,10,20],
        'n_estimators': [5, 10, 20, 50],
        'objective':['binary:logistic'],
        'seed': [1],
        'verbosity': [1]
    }

    xgb_clf = xgb.XGBClassifier()

    d_clf_cv = GridSearchCV(estimator=d_clf, param_grid=d_param_grid, cv=5, scoring='roc_auc')
    d_clf_cv.fit(X_train, y_train)

    logging.info("Decision tree optimised")

    lr_clf_cv = GridSearchCV(estimator=lr_clf, param_grid=lr_param_grid, cv=5, scoring='roc_auc')
    lr_clf_cv.fit(X_train, y_train)

    logging.info("Logistic regression optimised")

    rf_clf_cv = GridSearchCV(estimator=rf_clf, param_grid=rf_param_grid, cv=5, scoring='roc_auc')
    rf_clf_cv.fit(X_train, y_train)

    logging.info("Random Forest optimised")

    xgb_clf_cv = GridSearchCV(estimator=xgb_clf, param_grid=xgb_params, cv=5, scoring='roc_auc')
    xgb_clf_cv.fit(X_train, y_train)

    logging.info("xgboost classifier optimised")

    lr_best_params = lr_clf_cv.best_params_
    d_best_params = d_clf_cv.best_params_
    rf_best_params = rf_clf_cv.best_params_
    xgb_best_params = xgb_clf_cv.best_params_

    logging.info("Training the best models")
    lr_best_clf = LogisticRegression(**lr_best_params)
    d_best_clf = DecisionTreeClassifier(**d_best_params)
    rf_best_clf = RandomForestClassifier(**rf_best_params)
    xgb_best_clf = xgb.XGBClassifier(**xgb_best_params)

    lr_best_clf.fit(X_train, y_train)
    d_best_clf.fit(X_train, y_train)
    rf_best_clf.fit(X_train, y_train)
    xgb_best_clf.fit(X_train, y_train)

    return d_best_clf, lr_best_clf, rf_best_clf, xgb_best_clf



def select_best_model(d_tree_clf: DecisionTreeClassifier,
                      log_reg_clf: LogisticRegression,
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
    lr_roc_auc = evaluate_roc(log_reg_clf, X_val=X_test, y_val=y_test)
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
        },
        "logistic_regression" : {
            "model" : log_reg_clf,
            "roc_auc" : lr_roc_auc
        }
    }

    logging.info(f"Models and their best performance scores \n {model_performances}")

    best_model = sorted(model_performances.items(), reverse=True, key=lambda score: score[1]['roc_auc'])[0]

    return best_model


def create_bento(best_model: tuple, preprocessor: Pipeline, transformer: Pipeline):
    """Function to create bento"""

    if best_model[0] == 'xgboost':
        logging.info(f"Should use bentoml xgboost framework")
        
        model = best_model[1]['model']
        bentoml.xgboost.save_model(
        name=f'{MODEL_NAME}',
        model=model,
        custom_objects={
            "preprocessor": preprocessor,
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

    clean_strings_cols = [cols for cols in data if data[cols].dtype == 'object']

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
    preprocessed_data = preprocessor_pipeline.fit_transform(data)

    feature_cols = [cols for cols in preprocessed_data if cols not in [id_col, target]]
    cat_cols = [cols for cols in feature_cols if preprocessed_data[cols].dtype == 'object']
    X = preprocessed_data[feature_cols].copy()
    y = preprocessed_data[target].map({'y':1, 'n':0})

    logging.info("Splitting the data into train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    logging.info("Defining data transformation pipelines")
    transformer_pipeline = get_transformer_pipeline(cat_cols=cat_cols)
    
    logging.info("Running the transformer")
    transformed_data_train = transformer_pipeline.fit_transform(X_train, y_train)
    transformed_data_test = transformer_pipeline.transform(X_test)

    X_train_transformed = pd.DataFrame(data=transformed_data_train, columns=transformer_pipeline.get_feature_names_out())
    X_test_transformed = pd.DataFrame(data=transformed_data_test, columns=transformer_pipeline.get_feature_names_out())

    logging.info("Training the four models and hypertuning")
    d_tree_clf, log_reg_clf, rf_clf, xgb_clf = model_training(X_train=X_train_transformed, y_train=y_train)

    logging.info("Selecting the best model")
    best_model = select_best_model(d_tree_clf=d_tree_clf,
                                   log_reg_clf=log_reg_clf,
                                   xgb_clf=xgb_clf,
                                   rf_clf=rf_clf,
                                   X_test=X_test_transformed,
                                   y_test=y_test)


    logging.info("Preparing the bento")
    create_bento(best_model=best_model, preprocessor=preprocessor_pipeline, transformer=transformer_pipeline)
