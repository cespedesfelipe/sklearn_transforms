from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


# All sklearn Transforms must have the `transform` and `fit` methods
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        excluded_columns = [] if self.excluded_columns == None else self.excluded_columns
        df = data.drop(excluded_columns, axis='columns')
        si = SimpleImputer(
            missing_values=np.nan,
            strategy='mean' if self.strategy == None else self.strategy
        )
        si.fit(X=df)
        new_df = pd.DataFrame.from_records(
            data=si.transform(
                X=df
            ),
            columns=df.columns
        )
        for col in excluded_columns:
            new_df[col] = data[col]
        

        return new_df
