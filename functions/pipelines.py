
import pandas as pd
import numpy as np
import luigi
import warnings
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datetime import datetime, date, time, timedelta
from functions import reduce_mem_usage

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif

# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import f1_score, classification_report, plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)

new_features_list = ['interval']
#%%
class FeaturesGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, features_list):
        self.features_list = features_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # assert isinstance(X, pd.DataFrame)

        try:
            if 'interval' in self.features_list:
                X['interval'] = X['buy_time_y'] - X['buy_time_x']

            return X
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)

num_pipe = Pipeline([
    ('ncs', ColumnSelector(columns=numeric_features)),
    ('nsi', SimpleImputer(strategy="mean")),
    ('nss', StandardScaler()),
])

cat_pipe = Pipeline([
    ('ccs', ColumnSelector(columns=categorical_features)),
    ('csi', SimpleImputer(strategy="most_frequent")),
    ('coe', OneHotEncoder(handle_unknown='ignore')),
])

bool_pipe = Pipeline([
    ('bcs', ColumnSelector(columns=boolean_features)),
    ('bsi', SimpleImputer(strategy="most_frequent")),
])

transformer_list = [('num_pipe', num_pipe), ('cat_pipe', cat_pipe), ('bool_pipe', bool_pipe)]

transform_pipe = Pipeline([
    ('cs', ColumnSelector(columns=features)),
    ('fg', FeaturesGenerator(features_list=['interval'])),
    ('fu', FeatureUnion(transformer_list=transformer_list)),
])


fs_pipe = make_pipeline(
    transform_pipe,
    SelectKBest(k=50, score_func=f_classif),
    # SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_STATE), threshold=1e-3),
)