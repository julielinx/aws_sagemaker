import pandas as pd
import numpy as np

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import joblib
import json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFE

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

label_column = 'nbr_mobile_home_policies'
cat_cols = ['zip_agg_customer_subtype', 'zip_agg_customer_main_type']
INPUT_FEATURES_SIZE = 85

class OneHotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols):
        self._cat_cols = cat_cols
        self._col_names = None
        self._encoder = None
        self._all_cols = None

    def fit(self, df, y=None):
        print('Category cols:', self._cat_cols)
        self._encoder = OneHotEncoder()
        self._encoder = self._encoder.fit(df.loc[:, self._cat_cols])
        col_names = self._encoder.get_feature_names()
        self._col_names = [x.replace('x0', self._cat_cols[0]).replace('x1', self._cat_cols[1]) for x in col_names]
        print('Col names:', self._col_names)
        print('Successfully fit OneHot')
        return self

    def transform(self, df, y=None):
        sk_one_hot = self._encoder.transform(df.loc[:, self._cat_cols]).toarray()
        sk_one_hot = pd.DataFrame(sk_one_hot, columns=self._col_names)
        print('Dataframe sample:', sk_one_hot.head())
        sk_one_hot.set_index(df.index, inplace=True)
        sk_transformed = pd.concat([df, sk_one_hot], axis=1).drop(labels=self._cat_cols, axis=1)
        self._all_cols = sk_transformed.columns
        return self, sk_transformed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    args = parser.parse_args()
    

    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [ pd.read_csv(file) for file in input_files ]
    concat_data = pd.concat(raw_data)
    
    number_of_columns_x = concat_data.shape[1]
    train_y = concat_data.iloc[:,number_of_columns_x-1]
    train_X = concat_data.iloc[:,:number_of_columns_x-1]
        
    feature_selection_pipe = Pipeline([
        ('one_hot', OneHotTransformer(cat_cols=['zip_agg_customer_subtype', 'zip_agg_customer_main_type'])),
        ('svr', RFE(SVR(kernel="linear"))),# default: eliminate 50%
        ('f_reg',SelectKBest(f_regression, k=30)),
        ('mut_info',SelectKBest(mutual_info_regression, k=10))
    ])
    
    feature_selection_pipe.fit(train_X, train_y)

    joblib.dump(feature_selection_pipe, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")

    print('OneHot feature names:', feature_selection_pipe.named_steps['one_hot']._all_cols)
    print('SVR feature names:', feature_selection_pipe.named_steps['svr'].get_support())
    print('f_reg feature names:', feature_selection_pipe.named_steps['f_reg'].get_support())
    print('mut_info feature names:', feature_selection_pipe.named_steps['mut_info'].get_support())
    
    # feature_names = feature_selection_pipe.named_steps['one_hot']._all_cols
    # feature_names = feature_names[feature_selection_pipe.named_steps['svr'].get_support()]
    # feature_names = feature_names[feature_selection_pipe.named_steps['f_reg'].get_support()]
    # feature_names = feature_names[feature_selection_pipe.named_steps['mut_info'].get_support()]
    # joblib.dump(feature_names, os.path.join(args.model_dir, "selected_feature_names.joblib"))
    
    # print("Selected features are: {}".format(feature_names))
    
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
    
    print('Running predict_function')
    
    print('Input data shape at predict_fn: {}'.format(input_data.shape))
    if input_data.shape[1] == INPUT_FEATURES_SIZE:
        features = model.transform(input_data)
        return features
    elif input_data.shape[1] -- INPUT_FEATURES_SIZE + 1:
        features = model.transform(inputdata.iloc[:, :INPUT_FEATURES_SIZE])
        return np.insert(features, 0, input_data[label_column], axis=1)
    
def model_fn(model_dir):
    '''Deserialize fitted model'''
    
    print('Running model function')
    
    preprocessor = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return processor
