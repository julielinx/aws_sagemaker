import subprocess
import sys

def upgrade(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package, '--upgrade'])
    
upgrade('pandas==1.3.5')
upgrade('numpy')
upgrade('pyarrow')

import numpy as np
import pandas as pd
import boto3
# import logging
import os
import warnings
import joblib
import argparse

env_parser = argparse.ArgumentParser()
env_parser.add_argument('--INPUT_FEATURES_SIZE', type=int, dest='INPUT_FEATURES_SIZE')
    
env_args = env_parser.parse_args()
INPUT_FEATURES_SIZE = env_args.INPUT_FEATURES_SIZE

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.simplefilter("once")

true_false = ['true_false']
one_hot = ['one_hot']
date_cols = ['dates']
float_cols = ['floats']
max_of_list = ['max_of_list']
count_unique = ['nunique_of_list']
desc_stat_cols = ['desc_stats']
list_to_labels = ['multi_label']
drop_cols = ['random_col']

class TrueFalseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._col_names = None

    def fit(self, X, y=None):
        self._col_names = list(X.columns)
        return self

    def transform(self, X, y=None):
        print('Running TrueFalseTransformer')
        X.fillna('-1', inplace=True)
        X = X.replace({'true':'1', 'false':'0'})
        X = X.apply(pd.to_numeric, args=('coerce',))
        return X

    def get_feature_names(self):
        return self._col_names

class OneHotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._filler = 'ml_empty'
        self._col_names = None
        self._encoder = None
        self._transformer = None
        self._transformed_feats = []

    def fit(self, X, y=None):
        self._col_names = X.dropna(axis=1, how='all').columns
        X = X[self._col_names].fillna(self._filler)
        self._encoder = OneHotEncoder(handle_unknown='ignore')
        self._transformer = self._encoder.fit(X)
        self._transformed_feats = self._transformer.get_feature_names_out()
        return self

    def transform(self, X, y=None):
        print('Running OneHotTransformer')
        X = self._transformer.transform(X[self._col_names])
        return X

    def get_feature_names(self):
        return self._transformed_feats

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._col_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Running DateTransformer')
        temp_df = pd.DataFrame(index=X.index.copy())

        for col in X.columns:
            X[col] = pd.to_datetime(X[col])
            temp_df[f'{col}-month'] = X[col].dt.month.astype(float)
            temp_df[f'{col}-day_of_week'] = X[col].dt.dayofweek.astype(float)
            temp_df[f'{col}-hour'] = X[col].dt.hour.astype(float)
            temp_df[f'{col}-day_of_month'] = X[col].dt.day.astype(float)
            temp_df[f'{col}-is_month_start'] = X[col].dt.is_month_start.astype(int)
            temp_df[f'{col}-is_month_end'] = X[col].dt.is_month_end.astype(int)
        self._col_names = list(temp_df.columns)
        return temp_df

    def get_feature_names(self):
        return self._col_names

class FloatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._col_names = None

    def fit(self, X, y=None):
        self._col_names = list(X.columns)
        return self

    def transform(self, X, y=None):
        print('Running FloatTransformer')
        for col in self._col_names:
            if X[col].dtype == 'string':
                X[col] = X[col].astype(float)
        return X

    def get_feature_names(self):
        return self._col_names

class ListMaxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._col_names = None

    def fit(self, X, y=None):
        self._col_names = list(X.columns)
        return self

    def transform(self, X, y=None):
        print('Running ListMaxTransformer')
        temp_df = pd.DataFrame(index=X.index)
        for col in self._col_names:
            if X[col].dtype == 'string':
                X[col].fillna('-1', inplace=True)
                X[col] = X[col].str.split(pat=',').apply(set).apply(list)
            temp_series = X[col].explode()
            temp_series = temp_series.replace({'true':'1', 'false':'0'}).fillna('-1').apply(pd.to_numeric, args=('coerce',))
            temp_series = temp_series.groupby(temp_series.index).max()
            temp_df = temp_df.merge(temp_series, left_index=True, right_index=True, how='outer')
        temp_df = temp_df.fillna(0)
        return temp_df

    def get_feature_names(self):
        return self._col_names

class ListNuniqueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._col_names = None

    def fit(self, X, y=None):
        self._col_names = list(X.columns)
        return self

    def transform(self, X, y=None):
        print('Running ListNuniqueTransformer')
        temp_df = pd.DataFrame(index=X.index)
        for col in self._col_names:
            if X[col].dtype == 'string':
                X[col] = X[col].dropna().str.split(pat=',').apply(set).apply(list)
            temp_series = X[col].explode()
            temp_series = temp_series.groupby(temp_series.index).nunique()
            temp_df = temp_df.merge(temp_series, left_index=True, right_index=True, how='outer')
        temp_df = temp_df.fillna(0)
        return temp_df

    def get_feature_names(self):
        return self._col_names

class DescStatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._col_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Running DescStatTransformer')
        temp_df = pd.DataFrame(index=X.index)
        for col in X.columns:
            if X[col].dtype == 'string':
                X[col].fillna('-1', inplace=True)
                X[col] = X[col].str.split(pat=',').apply(set).apply(list)
            temp_series = X[col].explode()
            temp_series = temp_series.fillna('-1').apply(pd.to_numeric, args=('coerce',))
            temp_series = temp_series.groupby(temp_series.index).agg(['min', 'max', 'mean', 'std', 'nunique'])
            temp_series.columns = [f'{col}-{x}' for x in temp_series.columns]
            temp_df = temp_df.merge(temp_series, left_index=True, right_index=True, how='outer')
        temp_df = temp_df.fillna(0)
        self._col_names = list(temp_df.columns)
        return temp_df

    def get_feature_names(self):
        return self._col_names

class MultilabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._filler = 'ml_empty'
        self._encoder = None
        self._transformer = None
        self._col_names = None

    def fit(self, X, y=None):
        X = X.fillna(self._filler).str.split(pat=',').apply(set).apply(list)
        self._encoder = MultiLabelBinarizer()
        self._encoder.fit(X)
        self._col_names = [X.name + '_' + x for x in self._encoder.classes_]
        return self

    def transform(self, X, y=None):
        print('Running MultilabelTransformer')
        X = X.fillna(self._filler).str.split(pat=',').apply(set).apply(list)
        trans_array = self._encoder.transform(X)
        df = pd.DataFrame(trans_array, columns=self._col_names, index=X.index)        
        return df

    def get_feature_names(self):
        return self._col_names
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()
    
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    file_types = [x.split('.')[-1] for x in os.listdir(args.train)]
    type_list = list(set(file_types))
    if len(type_list) != 1:
           raise ValueError(('There are multiple file types or no files in {}.\n' +
                             'Please submit a single file or single file type.\n' +
                             'Accepted file types are csv, parquet, and json.').format(args.train))
    else:
        file_type = ''.join(type_list)
        if file_type == 'csv':
            raw_data = [ pd.read_csv(file) for file in input_files ]
        elif file_type == 'parquet':
            raw_data = [ pd.read_parquet(file) for file in input_files ]
        elif file_type == 'json':
            raw_data = [ pd.read_json(file) for file in input_files ]
        else:
            print('File type {} not accepted. Please use csv, parquet, or json'.format(file_type))
    
    train_data = pd.concat(raw_data)
    
    # print(train_data.head())
    
    preprocessor = ColumnTransformer([
        ('drop_cols', 'drop', drop_cols),
        ('truefalse', TrueFalseTransformer(), true_false),
        ('onehot', OneHotTransformer(), one_hot),
        ('dates', DateTransformer(), date_cols),
        ('floats', FloatTransformer(), float_cols),
        ('listmax', ListMaxTransformer(), max_of_list),
        ('nunique', ListNuniqueTransformer(), count_unique),
        ('descstats', DescStatTransformer(), desc_stat_cols),
        ('multilabel', MultilabelTransformer(), 'multi_label')],
        remainder='passthrough')

    print('Preprocessing data')
    preprocessor.fit(train_data)

    print('Saving preprocessor joblib')
    encoder_name = 'preprocessor.joblib'
    joblib.dump(preprocessor, os.path.join(args.model_dir, encoder_name))
    
    print('Defining and saving selected feature names')
    transform_col_list = drop_cols + true_false + one_hot + date_cols + float_cols + max_of_list + count_unique + desc_stat_cols + list_to_labels

    step_list = ['truefalse',
                 'onehot',
                 'dates',
                 'floats',
                 'listmax',
                 'nunique',
                 'descstats',
                 'multilabel']
    
    feature_names = []
    
    for step in step_list:
        print(step)
        item = preprocessor.named_transformers_[step].get_feature_names()
        if type(item) == list:
            feature_names = feature_names + item
        elif type(item) == str:
            feature_names = feature_names + [item]
        else:
            print('get_feature_names produced something other than a list or string')
            print(type(item))
            
    remainder_cols = list(train_data.drop(transform_col_list, axis=1).columns)
    feature_names = feature_names + remainder_cols
    
    joblib.dump(feature_names, os.path.join(args.model_dir, "feature_names.joblib"))

    print("Selected features are: {}".format(feature_names))
    
    
def input_fn(input_data, content_type):
    '''Parse input data payload
    
    Accepts csv, parquet, or json file types'''
    
    print('Running input function')
    
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data))
        return df
    elif content_type == 'application/x-parquet':
        df = pd.read_parquet(input_data)
    elif content_type == 'application/json':
        df = pd.read_json(input_data)
    else:
        raise ValueError("{} not supported by script".format(content_type))
        
def output_fn(prediction, accept):
    '''Format prediction output.
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    '''
    
    print('Running output function')
    
    if accept == 'application/json':
        instances = []
        for row in prediction.tolist():
            instances.append({'features': row})
            
        json_output = {'instances': instances}
        
        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException('{} accept type is not supported by this script')
        
def predict_fn(input_data, model):
    '''Preprocess input data
    
    The default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    '''
    
    feat_names = joblib.load(os.path.join(model_dir, 'selected_feature_names.joblib'))
    INPUT_FEATURES_SIZE = len(feat_names)
    
    print('Running predict_function')
    
    print('Input data shape at predict_fn: {}'.format(input_data.shape))
    if input_data.shape[1] == INPUT_FEATURES_SIZE:
        features = model.transform(input_data)
        return features
    elif input_data.shape[1] == INPUT_FEATURES_SIZE + 1:
        # this assumes the target is the last column
        features = model.transform(input_data.iloc[:, :-1])
        # # This assumes the target is the first column
        # features = model.transform(input_data.iloc[:, 1:])
        return np.insert(features, 0, input_data[label_column], axis=1)
        # What format should this be in? csv, json?
        # Should I add the column names here?
    
def model_fn(model_dir):
    '''Deserialize fitted model'''
    
    print('Running model function')
    
    preprocessor = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return preprocessor
