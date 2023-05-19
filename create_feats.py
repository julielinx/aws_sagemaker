import os
import pandas as pd
import numpy as np
import datetime
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample-size', type=str, dest='sample_size')
parser.add_argument('--group', type=str, dest='group')
args = parser.parse_args()
sample_ct = int(args.sample_size)

if __name__ == '__main__':
    
    output_path = '/opt/ml/processing/output'
    
    try:
        os.makedirs(os.path.join(output_path, "data"))
    except:
        pass

    # single value observations
    tf_col = []
    onehot_col = []
    float_col = []
    drop_col = []
    xrand_col = []
    group_col = []

    tf_vals = ['true', 'false', np.nan, '1', '0']
    onehot_vals = [np.nan, 'red', 'orange', 'yellow', 'green', 'blue', 'purple']
    float_vals = list(range(0,10)) + [x/10 for x in range(0, 100, 5)] +[np.nan]
    drop_vals = [np.nan] + list(range(0,10))
    xrand_vals = list(range(5))
    group_vals = ['first', 'second', 'third']

    col_list = zip([tf_col, onehot_col, float_col, drop_col, xrand_col, group_col],
                   [tf_vals, onehot_vals, float_vals, drop_vals, xrand_vals, group_vals])

    for col, vals in col_list:
        for _ in range(sample_ct):
            col.append(random.choice(vals))
        
    # date observations
    date_col = []

    for _ in range(sample_ct):
        try:
            date = datetime.date(2022, random.randint(1, 12), random.randint(1, 31))
            date_col.append(date)
        except ValueError:
            date_col.append(np.nan)

    # multivalue observations
    nbr_vals = list(range(0,10))
    str_vals = ['apple', 'orange', 'grape', 'pineapple', 'strawberry', 'blueberry', 'grapefruit', 'apple']

    nunique_col = []

    for _ in range(sample_ct):
        val_size = random.randint(0,6)
        if val_size < 1:
            nunique_col.append(np.nan)
        else:
            if random.randint(0,10) < 5:
                val_type = str_vals
            else:
                val_type = [str(x) for x in nbr_vals]
            val = random.choices(val_type,k=val_size)
            strified = ','.join(val)
            nunique_col.append(strified)

    descstat_col = []
    max_col = []

    nbrlst_cols = [descstat_col, max_col]

    for col in nbrlst_cols:
        for _ in range(sample_ct):
            val_size = random.randint(0,6)
            if val_size < 1:
                col.append(np.nan)
            else:
                val_type = [str(x) for x in nbr_vals]
                val = random.choices(val_type,k=val_size)
                strified = ','.join(val)
                col.append(strified)

    multi_col = []

    for _ in range(sample_ct):
        val_size = random.randint(0,6)
        if val_size < 1:
            multi_col.append(np.nan)
        else:
            val = random.choices(str_vals, k=val_size)
            strified = ','.join(val)
            multi_col.append(strified)

        # create dataframe
    sample_df = pd.DataFrame({
        'true_false':tf_col,
        'one_hot':onehot_col,
        'dates':date_col,
        'floats':float_col,
        'max_of_list':max_col,
        'nunique_of_list':nunique_col,
        'desc_stats':descstat_col,
        'multi_label':multi_col,
        'random_col':drop_col,
        'other':xrand_col,
        'group':group_col})
    print(f'Dataset provided sample size: {sample_ct}')
    print(f'Full dataframe shape: {sample_df.shape}')
    filtered_df = sample_df[sample_df['group']==args.group]
    print(f'Dataframe shape after being filtered by its group: {filtered_df.shape}')
    filtered_df.to_parquet(os.path.join(output_path, 'feats.parquet'), index=False)
