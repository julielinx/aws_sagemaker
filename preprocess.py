
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
def upgrade(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package, '--upgrade'])
    
upgrade('pandas==1.3.5')
upgrade('numpy')
install('category_encoders')

import pandas as pd
import numpy as np
import category_encoders as ce
import joblib
import os

if __name__ == '__main__':
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'
 
    try:
        os.makedirs(os.path.join(output_path, "train"))
        os.makedirs(os.path.join(output_path, "validate"))
        os.makedirs(os.path.join(output_path, "test"))
        os.makedirs(os.path.join(output_path, 'encoder'))
    except:
        pass
    
    cat_cols = ['zip_agg Customer Subtype', 'zip_agg Customer main type']

    ori_df = pd.read_csv(os.path.join(input_path, 'full_data.csv'))
    df = pd.DataFrame(ori_df['Nbr mobile home policies']).merge(ori_df.drop('Nbr mobile home policies', axis=1), left_index=True, right_index=True)
    print('Preprocessing data')
    encoder = ce.OneHotEncoder(cols=cat_cols, use_cat_names=True, handle_missing='return_nan')

    train_data, validation_data, test_data = np.split(
        df.sample(frac=1, random_state=1729),
        [int(0.7 * len(df)), int(0.9 * len(df))],)
    
    train_data = encoder.fit_transform(train_data)
    validation_data = encoder.transform(validation_data)
    test_data = encoder.transform(test_data)
    
    print('Saving dataframe')
    train_data.to_csv(os.path.join(output_path, 'train', 'train_feats.csv'), index=False)
    validation_data.to_csv(os.path.join(output_path, 'validate', 'validate_feats.csv'), index=False)
    test_data.to_csv(os.path.join(output_path, 'test', 'test_feats.csv'), index=False)
                              
    print('Saving preprocessor joblib')
    encoder_name = 'preprocessor.joblib'
    joblib.dump(encoder, os.path.join(output_path, 'encoder', encoder_name))
