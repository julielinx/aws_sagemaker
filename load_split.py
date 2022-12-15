import pandas as pd
import numpy as np
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
    
    print('Read in data')
    train = pd.read_csv(os.path.join(input_path, 'train.csv'))
    test = pd.read_csv(os.path.join(input_path, 'test.csv'))
    ground_truth = pd.read_csv(os.path.join(input_path, 'gt.csv'))
    columns = pd.read_csv(os.path.join(input_path, 'col_info.csv'))
    
    print('Getting column names')
    col_name_df = columns.iloc[1:87, 0].str.split(n=2, expand=True)
    col_name_df.columns = columns.iloc[0, 0].split(maxsplit=2)
    col_names = col_name_df['Name'].to_list()
    print(col_names)

    print('Combining data')
    test_df = pd.concat([test, ground_truth], axis=1)
    test_df.columns = col_names
    print(test_df.head())
    train.columns = col_names
    print(train.head())
    combined_df = pd.concat([train, test_df], ignore_index=True)
    print(combined_df.head())
#     df = pd.DataFrame(combined_df.iloc[:, -1]).merge(combined_df.iloc[:,0:-1], axis=1)
    df = pd.DataFrame(combined_df['CARAVAN']).merge(combined_df.drop('CARAVAN', axis=1), left_index=True, right_index=True)
    print(df.head())
    
    print('Splitting data')
    train_data, validation_data, test_data = np.split(
        df.sample(frac=1, random_state=1729),
        [int(0.7 * len(df)), int(0.9 * len(df))],)
    
    print('Saving dataframe')
    train_data.to_csv(os.path.join(output_path, 'train', 'train_feats.csv'), index=False)
    validation_data.to_csv(os.path.join(output_path, 'validate', 'validate_feats.csv'), index=False)
    test_data.to_csv(os.path.join(output_path, 'test', 'test_feats.csv'), index=False)
