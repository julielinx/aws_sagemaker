
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

    df = pd.read_csv(os.path.join(input_path, 'full_data.csv'))
    print('Preprocessing data')
    encoder = ce.OneHotEncoder(cols=cat_cols, use_cat_names=True, handle_missing='return_nan')
    processed_df = encoder.fit_transform(df)

    train_data, validation_data, test_data = np.split(
        processed_df.sample(frac=1, random_state=1729),
        [int(0.7 * len(processed_df)), int(0.9 * len(processed_df))],)
    
    print('Saving dataframe')
    train_data.to_csv(os.path.join(output_path, 'train', 'train_feats.csv'))
    validation_data.to_csv(os.path.join(output_path, 'validate', 'validate_feats.csv'))
    test_data.to_csv(os.path.join(output_path, 'test', 'test_feats.csv'))
                              
    print('Saving preprocessor joblib')
    encoder_name = 'preprocessor.joblib'
    joblib.dump(encoder, os.path.join(output_path, 'encoder', encoder_name))
