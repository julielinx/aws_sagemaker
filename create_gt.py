import os
import pandas as pd
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample-size', type=str, dest='sample_size')
parser.add_argument('--target', type=str, dest='target_col')
args = parser.parse_args()
sample_ct = int(args.sample_size)

if __name__ == '__main__':
    
    output_path = '/opt/ml/processing/output'
    
    try:
        os.makedirs(os.path.join(output_path, "data"))
    except:
        pass

    # single value observations
    target_col = []
    target_vals = [0,1]

    col_list = zip([target_col],
                   [target_vals])

    for col, vals in col_list:
        for _ in range(sample_ct):
            col.append(random.choice(vals))

    gt_df = pd.DataFrame({
        args.target_col:target_col})
    print(f'Ground truth provided sample size: {sample_ct}')
    print(f'Ground truth dataframe shape: {gt_df.shape}')
    gt_df.to_parquet(os.path.join(output_path, 'gt.parquet'), index=False)
