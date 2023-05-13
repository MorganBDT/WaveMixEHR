import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import pysiology # reference https://github.com/Gabrock94/Pysiology
import neurokit2 as nk #  https://neuropsychology.github.io/NeuroKit/functions/ecg.html # type: ignore 
from features.feature_extractor import Features # local library
import warnings
warnings.filterwarnings('ignore')

def read_signals( pth, waves=['II'] ): 
    signals = {w:[] for w in waves}
    pth = pth.replace('1hr_struc_train', '1hr_wf_train')
    with open(pth,'rb') as f: 
        data = pickle.load(f)
    for k in data.keys(): 
        for w in waves: 
            signals[w].append( data[k][w] )
    return signals

def get_features(ecg, sampling_rate): 
    extractor = Features(fs=sampling_rate)
    f1 = extractor.extract_features(ecg, filter_bandwidth=[3, 45], normalize=True, polarity_check=True, template_before=0.2, template_after=0.4)
    f2 = pysiology.electrocardiography.analyzeECG(ecg, sampling_rate, pnn50pnn20=False, freqAnalysis=False, freqAnalysisFiltered=False)
    neurokit_df, _ = nk.ecg_process(ecg, sampling_rate=sampling_rate)
    f3 = nk.ecg_analyze(neurokit_df, sampling_rate=sampling_rate).to_dict(orient='records')[0]
    features =  {**f1, **f2, **f3}
    return features

traj = pd.read_csv('../data/trajectories.csv')
waves = ['II'] # ignore 'I', 'III', 'V', 'PAP'
for idx in tqdm(traj.index): 
    s = read_signals(traj.savepath.iloc[idx], waves=['II'])['II']
    hourly_features = []
    for hr in range(12): 
        if not np.isnan(s[hr]).any() and len(set(s[hr]))>1: 
            f = None 
            try: 
                f = get_features(s[hr], 125)
            except Exception as e: 
                print('error on idx {} hr {}'.format(idx,hr))
            if f is not None: 
                hourly_features.append( f )
    traj_features = pd.DataFrame(hourly_features).mean().to_dict()
    for k,v in traj_features.items(): 
        traj.loc[idx,k] = v
    if not idx % 100:
        traj.to_csv('../data/trajectories_with_features.csv',index=False)
        print('saved at {}'.format(idx))