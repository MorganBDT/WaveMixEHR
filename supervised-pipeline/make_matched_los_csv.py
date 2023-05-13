import pdb

import pandas as pd
from datetime import datetime
import numpy as np

def main(ecg_path, csv, label_cols):
    """
    pair ECG features with binarized csv's for LOS and mortality
    """
    df = pd.read_csv(ecg_path)

    # in reality i probably want to add all the columns and just reorder so the hadm_ids match with df
    label_df = pd.read_csv(f'../data/{csv}.csv')
    label_df = label_df.rename(columns={'HADM_ID': 'hadm_id'})
    label_df = label_df.sort_values(by=['hadm_id'])
    df = df.sort_values(by=['hadm_id'])
    label_df = make_label_df(df, label_df, label_cols=label_cols['los'])
    label_df.to_csv(f'../data/labels_matched.csv', index=False)


def make_label_df(df, los_df, label_cols):
    """
    create LOS and mort df according to matching hadm_id
    if multiple matches in los_df, return the max length
    if day_cutoff, also add a binarized result
    """
    # get unique hadm_ids
    hadm_ids = los_df['hadm_id'].unique()
    all_labels = {
        'hadm_id': hadm_ids,
        'los': np.zeros((len(hadm_ids))),
        'mort': np.zeros((len(hadm_ids)))
    }
    for i, hadm_id in enumerate(hadm_ids):
        hadm_row = los_df[los_df['hadm_id'] == hadm_id]
        all_labels['los'][i] = compute_los(hadm_row)
        all_labels['mort'][i] = np.any([hadm['HOSPITAL_EXPIRE_FLAG'] for i, hadm in hadm_row.iterrows()])
    label_df = pd.DataFrame(all_labels)
    return label_df


def compute_los(df_subset):
    """
    from a single hadm_id but potentially multiple rows, compute the length of stay
    from the columns ADMITTIME and DISCHTIME
    return nan if empty subset
    """
    if len(df_subset) == 0:
        return np.nan
    admit_times = df_subset['ADMITTIME'].map(parse_datetime)
    disch_times = df_subset['DISCHTIME'].map(parse_datetime)
    lengths = (disch_times - admit_times).map(lambda x: x.days)
    return lengths.max()

def parse_datetime(dt_str):
    """
    get the datetime from the time str in the format 10/21/2008 23:09:00
    """
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

if __name__ == '__main__':
    ecg_path = '../data/filtered_ecg_features.csv'
    label_csv = 'mimic3/ADMISSIONS'

    label_cols = {
        'los': 'length_of_stay',
        'mort': 'HOSPITAL_EXPIRE_FLAG'
    }
    main(ecg_path, label_csv, label_cols)

