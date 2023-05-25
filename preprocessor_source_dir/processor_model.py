# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
# def upgrade(package):
#     subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", "-U", package])

# upgrade('scikit-learn==1.2.2')

import os
import warnings
import argparse
import csv
import json
import joblib
import pandas as pd
from io import StringIO
import ast
import traceback

from transformers import TrueFalseTransformer
from transformers import OneHotTransformer
from transformers import DateTransformer
from transformers import FloatTransformer
from transformers import ListMaxTransformer
from transformers import ListNuniqueTransformer
from transformers import DescStatTransformer
from transformers import MultilabelTransformer
from transformers import DropSingleValueCols
from transformers import RemoveCollinearity

from sklearn import set_config

from sagemaker_containers.beta.framework import(
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker)

warnings.simplefilter("once")
set_config(transform_output="pandas")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model', type=str, default=os.environ.get('SM_CHANNEL_INPUT_MODEL'))
    parser.add_argument('--feature-names', type=str, default=os.environ.get('SM_CHANNEL_FEATURE_NAMES'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()
    
    print('Args: {}'.format(args))
    
    print('Loading preprocessor')
    preprocessor = joblib.load(os.path.join(args.input_model, 'preprocessor.joblib'))
    print('Loading feature names')
    feat_names = joblib.load(os.path.join(args.feature_names, 'feature_names.joblib'))
    
    print('Saving models')
    # This is done so that it is all put into a tar.gz file that can be used at inference
    joblib.dump(preprocessor, os.path.join(args.model_dir, 'preprocessor.joblib'))
    joblib.dump(feat_names, os.path.join(args.model_dir, 'feature_names.joblib'))

def input_fn(input_data, content_type):
    '''Parse input data payload
    
    Accepts csv, parquet, or json file types'''
    
    print('Pandas version:', pd.__version__)
    # print('Sklearn version:', sklearn.__version__)
    
    print(f'Loading input data with content type {content_type}')
    print(f'Input data: {input_data}')
    print('Input data type', type(input_data))
    print('Content type:', content_type)
    
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data))
        print(df)
        # df = pd.read_csv(input_data)
    # elif content_type == 'application/x-parquet':
    #     df = pd.read_parquet(input_data)
    #     print(df)
    elif content_type == 'application/json':
        json_data = json.loads(input_data)
        print('json data object:', json_data)
        print('json data object type:', type(json_data))
        df = pd.DataFrame(json_data, index=[0,])
    else:
        raise ValueError("{} not supported by script".format(content_type))
    print(df)
    return df
        
def output_fn(prediction, accept):
    '''Format prediction output.
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    '''
    
    print(f'Running output function with accept type {accept}')
    feat_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
    print('Feature names:', feat_names)
    df = pd.DataFrame(prediction, columns=feat_names)
    print('Output data:', df.head())
    
    if accept == 'application/json':
        # instances = []
        # for row in prediction.tolist():
        #     instances.append({'features': row})
        # json_output = {'instances': instances}
        return worker.Response(df.to_json(orient='records'), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException(f'{accept} accept type is not supported by this script')
        
def predict_fn(input_data, preprocessor):
    '''Preprocess input data
    
    The default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    '''
    
    print('Preprocessing data')    
    print('Input data shape at predict_fn: {}'.format(input_data.shape))
    features = preprocessor.transform(input_data)
    print(f'Data shape after prediction: {features.shape}')
    print(features)
    return features        
    
def model_fn(model_dir):
    '''Deserialize fitted model'''
    
    print('Loading model')
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    return preprocessor
