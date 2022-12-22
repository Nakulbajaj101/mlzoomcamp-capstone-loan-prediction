from typing import List
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def clean_strings(val):
    """Function to make data values consistent"""

    return val.replace(" ","_").replace("-", "_")



class BinaryEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, column_list: str, value: int):
        """Instatiating the class"""

        self.column_list = column_list
        self.value = value


    def fit(self, X):
        """Custom fit function"""

        return self

    def transform(self, X):
        """Custom transformer that converts yes and no to 1 and 0"""
        
        for col in self.column_list:
            X[col] = X[col].apply(lambda val: val if str(val) == 'nan' else 1 if val.lower() == self.value else 0)
        return X


class CleanStrings(BaseEstimator, TransformerMixin):

    def __init__(self, column_list: List[str]):

        self.column_list = column_list


    def fit(self, X):
        """Custom fit function"""

        return self

    def transform(self, X):
        """Custom transformer to clean all strings and make them lower case"""

        for col in self.column_list:
            if X[col].dtype == 'object':
                X[col] = X[col].apply(lambda val: val.lower() if isinstance(val, str) else val)
                X[col] = X[col].apply(lambda val: clean_strings(val) if isinstance(val, str) else val)

        return X


class ColumnDropperTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,column_list: List[str]):

        self.column_list = column_list

    def fit(self, X, y=None):
        """Custom fit function"""

        return self 

    def transform(self,X,y=None):
        """Custom transformer to drop specific columns"""

        return X.drop(self.column_list,axis=1)


class CustomColumnMapping(BaseEstimator, TransformerMixin):

    def __init__(self, column_mapping: dict):

        self.column_mapping = column_mapping

    def fit(self, X, y=None):
        """Custom fit function"""

        return self 

    def transform(self,X,y=None) -> pd.DataFrame:
        """Custom transformer to rename columns"""

        X = X.rename(columns=self.column_mapping)

        return X


class MultiplyColumnValue(BaseEstimator, TransformerMixin):

    def __init__(self, column_list: List[str], multiply_by: float):

        self.column_list = column_list
        self.multiply_by = multiply_by

    def fit(self, X, y=None):
        """Custom fit function"""

        return self 

    def transform(self,X,y=None) -> pd.DataFrame:
        """Custom transformer to rename columns"""

        for col in self.column_list:
            X[col] = X[col]*self.multiply_by

        return X

class DivideColumnValue(BaseEstimator, TransformerMixin):

    def __init__(self, column_list: List[str], divide_by: float):

        self.column_list = column_list
        self.divide_by = divide_by

    def fit(self, X, y=None):
        """Custom fit function"""

        return self 

    def transform(self,X,y=None) -> pd.DataFrame:
        """Custom transformer to rename columns"""

        for col in self.column_list:
            X[col] = X[col]/self.divide_by

        return X


class CustomRowMapping(BaseEstimator, TransformerMixin):

    def __init__(self, column_list: List[str], column_value_mapping: dict):

        self.column_list = column_list
        self.column_value_mapping = column_value_mapping

    def fit(self, X, y=None):
        """Custom fit function"""

        return self 

    def transform(self, X, y=None) -> pd.DataFrame:
        """Custom transformer to update data values"""

        for col in self.column_list:
            X[col] = X[col].map(self.column_value_mapping[col])

        return X


class PrefixStringEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, column_list: List[str], string_val: str):

        self.column_list = column_list
        self.string_val = string_val

    def fit(self, X, y=None):
        """Custom fit function"""

        return self 

    def transform(self, X, y=None) -> pd.DataFrame:
        """Custom transformer to update data values"""

        for col in self.column_list:
            X[col] = X[col].apply(lambda val: f"{self.string_val}_{int(val/12)}" if str(val) != 'nan' else val)

        return X
