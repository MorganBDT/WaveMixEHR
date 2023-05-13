import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('../data/length_of_stay.csv')
    exclude_cols = ['subject_id', 'hadm_id', 'icustay_id', 'length_of_stay']
    include_cols = [col for col in df.columns if col not in exclude_cols]

    # for each column, discretize into 10 bins according to quantiles
    for col in include_cols:
        df[col] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')

    # save to csv
    df.to_csv('../data/length_of_stay_discretized.csv', index=False)
