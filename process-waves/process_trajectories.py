import os 
import pandas as pd
import numpy as np 
import pickle

repo_pth = '/storage/payal/dropbox/private/wave-mixehr/'
traj_pth = '/storage/shared/trajectories/mimic_iii/'
waves = ['II'] # ignore 'I', 'III', 'V', 'PAP'

# process trajectories
traj = pd.concat([
    pd.read_csv(traj_pth+'train_dataset.csv'),
    pd.read_csv(traj_pth+'tuning_dataset.csv'),
])
traj = traj.rename(columns={'mrn':'subject_id'})
print('patients',traj.get(['subject_id']).drop_duplicates().shape[0])
print('admissions',traj.get(['subject_id','hadm_id']).drop_duplicates().shape[0])
traj = traj.reset_index(drop=1)

# keep only trajectories of length 12
delete_index = []
for idx, savepth in enumerate(traj.savepath.values): 
    savepth = savepth.replace('1hr_struc_train', '1hr_wf_train')
    with open(savepth,'rb') as f: 
        data = pickle.load(f)
    if len(data.keys()) != 12: 
        delete_index.append( idx )
traj = traj.query('index not in @delete_index')
traj = traj.reset_index(drop=1)

# measure missingness of waveforms at each hour
for idx in traj.index: 
    p = traj.savepath.loc[idx].replace('1hr_struc_train', '1hr_wf_train')
    with open(p,'rb') as f: 
        data = pickle.load(f)

    for hr, k in enumerate(data.keys()): 
        hr += 1
        for wave in waves:
            if isinstance(data[k][wave],np.ndarray): missing=np.isnan(data[k][wave]).any()
            else: missing=True
            traj.loc[idx,'wave_'+wave+'_hr_'+str(hr)] = int(not missing)

    for wave in waves:
        count = 0
        for hr in range(1,13):
            count += traj.loc[idx,'wave_'+wave+'_hr_'+str(hr)]
        traj.loc[idx,'wave_'+wave+'_count'] = count

# assign splits
traj = traj.reset_index(drop=1)
pt = list(set(traj.subject_id.values))
split = np.random.choice(a=['train','tune','test'],size=len(pt),p=[0.85,0.05,0.10])
split = pd.DataFrame([*zip(pt, split)], columns=['pt','split'])
for group in ['train','tune','test']:
    x = split.query('split==@group').pt.values
    traj.loc[traj.query('subject_id in @x').index.values, 'split'] = group

# save
traj = traj.reset_index(drop=1)
traj.to_csv(repo_pth+'data/trajectories.csv', index=False)
print('done')