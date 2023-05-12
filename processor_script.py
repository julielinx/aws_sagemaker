# import subprocess
# import sys

# def upgrade(package):
#     subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package, '--upgrade'])
    
# upgrade('pandas==1.3.5')
# upgrade('numpy')
# upgrade('pyarrow')

import numpy as np
import pandas as pd
import os
import warnings
import joblib
import argparse
from io import StringIO

from transformers import TrueFalseTransformer
from transformers import OneHotTransformer
from transformers import DateTransformer
from transformers import FloatTransformer
from transformers import ListMaxTransformer
from transformers import ListNuniqueTransformer
from transformers import DescStatTransformer
from transformers import MultilabelTransformer

from sagemaker_containers.beta.framework import (
    encoders, worker)

# env_parser = argparse.ArgumentParser()
# env_parser.add_argument('--INPUT_FEATURES_SIZE', type=int, dest='INPUT_FEATURES_SIZE')
# env_args = env_parser.parse_args()
# INPUT_FEATURES_SIZE = env_args.INPUT_FEATURES_SIZE
INPUT_FEATURES_SIZE = 10

# from sklearn.pipeline import Pipeline
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
            print(f'get_feature_names from {step} produced something other than a list or string')
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
        return df
    elif content_type == 'application/json':
        df = pd.read_json(input_data)
        return df
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
    
    # feat_names = joblib.load(os.path.join(model_dir, 'selected_feature_names.joblib'))
    # INPUT_FEATURES_SIZE = len(feat_names)
    
    print('Running predict_function')
    
    print('Input data shape at predict_fn: {}'.format(input_data.shape))
    if input_data.shape[1] == INPUT_FEATURES_SIZE:
        features = model.transform(input_data)
        print(f'Data shape after prediction: {features.shape}')
        print(features)
        return features
    elif input_data.shape[1] == INPUT_FEATURES_SIZE + 1:
        # this assumes the target is the last column
        features = model.transform(input_data.iloc[:, :-1])
        # # This assumes the target is the first column
        # features = model.transform(input_data.iloc[:, 1:])
        print(f'Data shape after prediction: {features.shape}')
        print(features)
        return np.insert(features, 0, input_data[label_column], axis=1)
        # What format should this be in? csv, json?
        # Should I add the column names here?
    
    
def model_fn(model_dir):
    '''Deserialize fitted model'''
    
    print('Running model function')
    
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    return preprocessor
