import numpy as np
import pandas as pd
import os
import warnings
import joblib
import argparse
from io import StringIO
import sys

sys.path.append('/opt/app/0_sample_version/davids_workflow/my_version/')

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

from sagemaker_containers.beta.framework import (
    encoders, worker)

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import set_config

warnings.simplefilter("once")
set_config(transform_output="pandas")

# print("This is a test and you passed")

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
    parser.add_argument('--target', type=str, dest='target_col')
    parser.add_argument('--train-size', type=str, dest='train_size', default='0.8')
    parser.add_argument('--file-format', type=str, dest='file_format', default='csv')
    args = parser.parse_args()
    
    train_size = float(args.train_size)
    if train_size > 1:
        train_size = train_size/100
    
    def splitTransform(df, transformer, target_col=args.target_col):
        x = df.drop(target_col, axis=1)
        y = df[[target_col]]
        feats = transformer.transform(x)
        return feats, y
    
    input_path = '/opt/ml/processing/input/data'
    output_path = '/opt/ml/processing/output'
    
    try:
        os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "validate"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test", 'feats'), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test", 'target'), exist_ok=True)
        os.makedirs(os.path.join(output_path, "encoder"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "encoder_cols"), exist_ok=True)
    except:
        pass
    
#     print('input_path:', os.listdir(input_path))
#     print('output_path folders:', os.listdir(output_path))
    
#     print('Pandas version:', pd.__version__)
#     print('Numpy version:', np.__version__)
#     print('SKLearn version:', sklearn.__version__)

    print('Loading data')
    feats_df = pd.read_parquet(os.path.join(input_path, 'feats', 'feats.parquet'))
    print(f'Feature data size: {feats_df.shape}')
    gt_df = pd.read_parquet(os.path.join(input_path, 'gt', 'gt.parquet'))
    print(f'Ground truth data size {gt_df.shape}')
    print('Combining features and ground truth')
    full_data = feats_df.merge(gt_df, right_index=True, left_index=True, how='inner')
    print(f'Merged dataframe size: {full_data.shape}')
    print(f'Set target as {args.target_col}')
    
    train_data, other = train_test_split(full_data, train_size=train_size, random_state=12, stratify=full_data[args.target_col])
    test_data, validate_data = train_test_split(other, train_size=0.5, random_state=12, stratify=other[args.target_col])
    
    print('Train data size:', train_data.shape)
    print('Validate data size:', validate_data.shape)
    print('Test data size:', test_data.shape)

    # Offload memory
    feats_df = None
    gt_df = None
    full_data = None
    
    # print(train_data.head())
        
    preprocessor = ColumnTransformer([
        ('truefalse', TrueFalseTransformer(), true_false),
        ('onehot', OneHotTransformer(), one_hot),
        ('dates', DateTransformer(), date_cols),
        ('floats', FloatTransformer(), float_cols),
        ('listmax', ListMaxTransformer(), max_of_list),
        ('nunique', ListNuniqueTransformer(), count_unique),
        ('descstats', DescStatTransformer(), desc_stat_cols),
        ('multilabel', MultilabelTransformer(), 'multi_label')],
        verbose_feature_names_out=False)

    extras = Pipeline([
        ('dropsingle', DropSingleValueCols()),
        ('removemulticollinear', RemoveCollinearity())])
    
    processor = Pipeline([
        ('preprocess', preprocessor),
        ('additional', extras)])

#     processor = ColumnTransformer([
#         ('truefalse', TrueFalseTransformer(), true_false),
#         ('onehot', OneHotTransformer(), one_hot),
#         ('dates', DateTransformer(), date_cols),
#         ('floats', FloatTransformer(), float_cols),
#         ('listmax', ListMaxTransformer(), max_of_list),
#         ('nunique', ListNuniqueTransformer(), count_unique),
#         ('descstats', DescStatTransformer(), desc_stat_cols),
#         ('multilabel', MultilabelTransformer(), 'multi_label')],
#         verbose_feature_names_out=False)


    print('Preprocessing data')
    processor.fit(train_data.drop(args.target_col, axis=1))
    
    print('Transforming data')
    train_feats, train_target = splitTransform(train_data, processor)
    validate_feats, validate_target = splitTransform(validate_data, processor)
    test_feats, test_target = splitTransform(test_data, processor)

    feature_names = list(train_feats.columns)
    print("Selected features are: {}".format(feature_names))

    print('Saving preprocessor and feature_name joblibs')
    joblib.dump(processor, os.path.join(output_path, 'encoder', 'preprocessor.joblib'))
    joblib.dump(feature_names, os.path.join(output_path, 'encoder_cols', 'feature_names.joblib'))
    
    print('Saving dataframes')
    pd.concat([train_target, train_feats], axis=1).to_csv(os.path.join(output_path, 'train', 'train.csv'), index=False)
    pd.concat([validate_target, validate_feats], axis=1).to_csv(os.path.join(output_path, 'validate', 'validate.csv'), index=False)
    # test.to_csv(os.path.join(output_path, 'test', 'feats', 'test.csv'), index=False)
    test_feats.to_csv(os.path.join(output_path, 'test', 'feats', 'test_x.csv'), index=False, header=False)
    test_target.to_csv(os.path.join(output_path, 'test', 'target', 'test_y.csv'), index=False, header=False)
