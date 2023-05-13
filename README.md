# Wave-MixEHR
A project that extends MixEHR to incorporate ECG waveforms
Original MixEHR repository: https://github.com/li-lab-mcgill/mixehr

# Data
To download raw data from MIMIC-III, obtain access privileges on PhysioNet and then use
```bash
wget -r -N -c -np https://physionet.org/files/mimic3wdb-matched/1.0/
wget -r -N -c -np --user <USERNAME> --ask-password https://physionet.org/files/mimiciii/1.4/
```

# Extracting waveform features
See scripts in process-waves directory

# Processing waveform features and other MIMIC-III modalities, and training MixEHR
Run data/mixehr/format_for_mixehr.ipynb. Or, to include only patients with 36 hour+ admissions and using only data from the first 24 hours, run data/mixehr/early_data_only/format_for_mixehr_early.ipynb. This will generate .txt files in the format required by the original MixEHR model, the source code of which is included here with no modifications. 

# Visualizing topic associations using word clouds
Run word-clouds/word_clouds.ipynb

# Generating t-SNE plots of patient distributions in topic-space, and PCA plots of topics in ECG feature association-space: 
Run data/tsne_pca.ipynb

# Training models to predict mortality and length of stay
See ipython notebooks and .py files in data/supervised_pipeline:
* Run topic_explore.ipynb to organize topics into features with hadm_ids and subj_ids. The last code block creates a binarized LOS and mortality csv according to MIMIC's admissions.csv
* Then run the main method of run_xgboost.py, which will save results to data/supervised_pipeline. The files are saved in nested dicts according to task, modality, and settings for the modality. These can be visualized with plot_results.ipynb
