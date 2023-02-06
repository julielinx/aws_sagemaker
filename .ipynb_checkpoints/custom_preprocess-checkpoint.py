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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--label_col', type=str, default='nbr_mobile_home_policies')
    args = parser.parse_args()
    
    cat_feats = ['zip_agg_customer_subtype', 'zip_agg_customer_main_type']
    nbr_cols = df.shape[1]
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
    
    col_transformer = ColumnTransformer([
            ('encoder', OneHotEncoder(), cat_feats)],
        remainder='passthrough')
        
    processed_df = col_transformer.fit(train_X, train_y)

    joblib.dump(feature_selection_pipe, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    one_hot_cols = col_transformer.named_transformers_['encoder'].get_feature_names()
    feature_names = []

    for i, col in enumerate(cat_feats):
        del_str = f'x{i}'
        col_list = [itm for itm in one_hot_cols if itm.startswith(del_str)]
        feature_names = feature_names + [x.replace(del_str, col) for x in col_list]
        
    feature_names = feature_names + list(train_X.drop(cat_feats, axis=1).columns)
    
    joblib.dump(feature_names, os.path.join(args.model_dir, "selected_feature_names.joblib"))
    
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
