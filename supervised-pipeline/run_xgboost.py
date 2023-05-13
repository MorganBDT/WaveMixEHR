import pdb
from sklearn.neighbors import KNeighborsClassifier
import itertools
import matplotlib.pyplot as plt
import pickle
import numpy as np
# import partial
from functools import partial
import pandas as pd
import torch
# XGBoost
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
# XGBBoost
import xgboost as xgb


def plot_performance(performance_df, task, metric):
    """
    Plots each metric for each task, where each color is a feature type, and each bar is a modality.

    Parameters:
    performance_df (pandas.DataFrame): the dataframe that contains the performance metrics
    task (str): the name of the task to plot
    metric (str): the name of the metric to plot

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, feature_type in enumerate(performance_df.columns.levels[2]):
        for j, modality in enumerate(performance_df.columns.levels[1]):
            metric_value = performance_df[task][modality][feature_type][metric]
            plt.bar(j + i*0.1, metric_value, width=0.1, color=colors[i], label=feature_type)

    plt.xticks(range(len(performance_df.columns.levels[1])), performance_df.columns.levels[1])
    plt.xlabel('Modality')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} for {task.capitalize()} Task')
    plt.legend(loc='best')
    plt.show()

def get_split(df, split):
    """
    get train or test split
    """
    return df[df['split'] == split]

def get_binary_labels(ids, los_df, cutoff, label_col):
    """
    get binary LOS labels from df and los_df. drop missing LOS rows
    Args:
        df: dataframe with hadm_ids and features, potential repeated hadm_ids
        los_df: dataframe with hadm_ids and los
        day_cutoff: int, day cutoff for LOS binarization
    Returns:
        labels: list of binary LOS labels corresponding to each row in df
    """
    los_dict = dict(zip(los_df['hadm_id'], los_df[label_col]))
    labels = []
    for hadm_id in ids:
        if hadm_id in los_dict:
            labels.append(los_dict[hadm_id] > cutoff)
        else:
            labels.append(np.nan)
    return labels




def main(df, normalize, **kwargs):
    train_set = get_split(df, 'train')
    val_set = get_split(df, 'val')
    train_set = pd.concat([train_set, val_set])
    test_set = get_split(df, 'test')
    train_set, train_labels, train_ids = process_split(train_set, **kwargs)
    test_set, test_labels, test_ids = process_split(test_set, **kwargs)
    if normalize:
        train_set = normalize_continuous(train_set)
        test_set = normalize_continuous(test_set)
    # XGBoost
    if kwargs['model'] == 'xgboost':
        clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=0)
        clf.fit(train_set, train_labels)
        preds = clf.predict_proba(test_set)[:, 1]
        train_preds = clf.predict_proba(train_set)[:, 1]
    elif kwargs['model'] == 'gradient_boosting':
        clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
                                         max_depth=None, random_state=0).fit(train_set, train_labels)

        clf.fit(train_set, train_labels)
        preds = clf.predict_proba(test_set)[:, 1]
    elif kwargs['model'] == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=10)
        clf.fit(train_set, train_labels)
        preds = clf.predict(test_set)[:, 1]


    auc = roc_auc_score(test_labels, preds)
    auprc = average_precision_score(test_labels, preds)
    print(f'\t AUC:: {auc:.3f}, AUPRC:: {auprc:.3f}')
    # find the optimal threshold on the training set and apply to testing set
    optimal_threshold = find_optimal_threshold(train_labels, train_preds)
    binary_preds = preds > optimal_threshold
    accuracy = accuracy_score(test_labels, binary_preds)
    f1 = f1_score(test_labels, binary_preds)
    precision = precision_score(test_labels, binary_preds)
    recall = recall_score(test_labels, binary_preds)
    # print all metrics
    print(f'Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}')
    return clf, preds, test_labels, accuracy, f1, precision, recall, auc, test_ids

def find_optimal_threshold(labels, preds):
    """
    get optimal threshold according to F1 for preds between 0 and 1
    """
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    for threshold in thresholds:
        binary_preds = preds > threshold
        f1_scores.append(f1_score(labels, binary_preds))
    return thresholds[np.argmax(f1_scores)]

def process_split(df, los_df, feat_colnames, label_colname, cutoff, missing_method='zero_fill', **kwargs):
    """
    get features, labels, and ids for split
    """
    df_ids = df['hadm_id']
    df = df.loc[:, feat_colnames]
    df = replace_missing_vals(df, method=missing_method)
    labels = get_binary_labels(df_ids, los_df, cutoff=cutoff, label_col=label_colname)
    df, labels = drop_missing_labels(df, labels)
    return df, list(labels), df_ids

def drop_missing_labels(df, labels):
    """
    remove labels with nan values and rows corresponding to those nan values
    Returns:
        df: dataframe with rows containing nan labels removed
        labels: list of labels with nan values removed
    """
    df['binary_label'] = labels
    df = drop_missing_rows(df)
    labels = df['binary_label']
    df = df.drop(columns=['binary_label'])
    return df, labels

# replace missing values with 0
def replace_missing_vals(df, method='zero_fill'):
    """
    either replace missing vals with zero or drop cols
    """
    if method == 'zero_fill':
        df = df.fillna(0)
        df.replace([np.inf, -np.inf], 0, inplace=True)
    elif method == 'drop':
        pass
    return df

def normalize_continuous(feature_df):
    """
    normalize and standardize continuous features between 0 and 1
    """
    feature_df = feature_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # normalize to mean 0 and std 1
    feature_df = feature_df.apply(lambda x: (x - x.mean()) / x.std())
    # replace nan values with 0
    feature_df = feature_df.fillna(0)
    return feature_df

def get_train_test_dfs(df):
    train_set = df[df['split'] == 'train']
    test_set = df[df['split'] == 'test']
    # drop split column
    train_set = train_set.drop(columns=['split'])
    test_set = test_set.drop(columns=['split'])
    return train_set, test_set

def get_labels(df, label_col):
    """
    get binary LOS labels and drop from df
    """
    labels = df['binary_label']
    df = df.drop(columns=['binary_label', label_col])
    return df, labels

def drop_missing_rows(df):
    """
    find rows with nan values and drop them
    """
    df = df.dropna()
    return df

def get_topic_file(name):
    return f'supervised_pipeline/{name}_topics.csv'

def plot_importance(model, feature_names, model_name, orientation='horizontal'):
    if model_name in ['xgboost', 'gradient_boosting']:
        importances = model.feature_importances_
        feature_names = np.array(feature_names)
        indices = np.argsort(importances)[::-1]
    elif model_name == 'KNN':
        importances = clf.feature_importances_[feature_names]
        feature_names = np.array(feature_names)[feature_names]
        indices = np.argsort(importances)[::-1]
    if orientation == 'vertical':
        shape = len(feature_names)
        # plot the feature importance
        plt.figure(figsize=(20, 50))
        plt.bar(range(shape), importances[indices], color="r", align="center")
        plt.xticks(indices, feature_names, rotation='vertical', fontsize=15)
        plt.xlim([-1, shape])
        plt.ylabel('Relative importance')
        plt.xlabel('Feature')
    if orientation == 'horizontal':
        # plot the feature importance
        plt.figure(figsize=(10, 10))
        # reverse order of indices, filter to top 25 most important features
        indices = indices[:12]
        indices = indices[::-1]
        shape = len(indices)
        plt.barh(range(shape), importances[indices], height=0.6, color="r", align="center")
        plt.yticks(range(shape), feature_names[indices], fontsize=20)
        plt.ylim([-1, shape])
        plt.xlabel('Relative importance', fontsize=10)
        plt.ylabel('Top Features', fontsize=20)


def get_dict_permutations(dict1, dict2):
    """
      Get all possible combinations of keys from two dictionaries.

      Args:
          dict1 (dict): A dictionary containing key-value pairs.
          dict2 (dict): Another dictionary containing key-value pairs.

      Returns:
          A list of tuples, where each tuple contains two keys, one from each dictionary.

      Example:
          dict1 = {'key1': 'value1', 'key2': 'value2'}
          dict2 = {'key3': 'value3', 'key4': 'value4', 'key5': 'value5'}
          key_combinations = get_key_combinations(dict1, dict2)
          print(key_combinations)  # [('key1', 'key3'), ('key1', 'key4'), ('key1', 'key5'), ('key2', 'key3'), ('key2', 'key4'), ('key2', 'key5')]

      """
    dict1_keys = list(dict1.keys())
    dict2_keys = list(dict2.keys())
    # Create a list of tuples containing all possible combinations of keys
    key_combinations = list(itertools.product(dict1_keys, dict2_keys))
    return key_combinations



def create_prediction_dict(ecg_preds, topic_preds, fusion_preds, ecg_ids, topic_ids, fusion_ids=None, ids_with_labels=None):
    """
    Creates a dictionary of tuples for each prediction according to the ids.

    Args:
        ecg_preds (list): List of ECG predictions.
        topic_preds (list): List of topic predictions.
        ecg_ids (list): List of ids used for the ECG modality.
        topic_ids (list): List of ids used for the topic modality.
        fusion_preds (list, optional): List of fusion predictions.
        fusion_ids (list, optional): List of ids used for the fusion modality.

    Returns:
        A dictionary of tuples for each prediction according to the ids.
    """
    # get the list of all ids
    fusion_ids = fusion_ids if fusion_ids else []
    all_ids = [ecg_ids, topic_ids, fusion_ids]
    all_preds = [ecg_preds, topic_preds, fusion_preds]
    all_ids = [item for sublist in all_ids for item in sublist]
    if ids_with_labels is not None:
        all_ids = [item for item in all_ids if item in ids_with_labels]
    shared_ids = np.unique(all_ids)
    prediction_dict = {id: [] for id in shared_ids}
    for pred_list in all_preds:
        for i in range(len(pred_list)):
            pred = pred_list[i]
            id = all_ids[i]
            prediction_dict[id].append(pred)
    return prediction_dict

def late_fusion(performance_tracker, ecg_type, topic_type, label_df, task):
    """
       Fuse predictions from two modalities (ECG and topics) for a given task using late fusion.

       Args:
           performance_tracker (dict): A nested dictionary containing performance metrics and predictions for
                                       each modality and task.
           ecg_type (str): The type of ECG features used for prediction (e.g., "raw", "spectrograms").
           topic_type (str): The type of topic model used for prediction (e.g., "LDA", "NMF").
           task (str): The prediction task (e.g., "los", "mort").

       Returns:
           dict: A dictionary containing predictions for each patient, with keys being the patient IDs and
                 values being tuples of ECG and topic predictions.
       """
    # get predictions for each modality
    # can even get the correlation between predictions with different methods! with xgboost (random foresting)
    # this is essentially getting one set of trees for topics only, one set for ecg only, and one set for both
    fusion_name = get_intermed_name(ecg_type, topic_type)
    ecg_preds = performance_tracker['ecg'][ecg_type]['preds']
    topics_preds = performance_tracker['topics'][topic_type]['preds']
    fusion_preds = performance_tracker['intermed_fusion'][fusion_name]['preds']
    # align according to hadm_id
    ecg_hadm_ids = performance_tracker['ecg'][ecg_type]['test_hadm_ids']
    topic_hadm_ids = performance_tracker['topics'][topic_type]['test_hadm_ids']
    fused_hadm_ids = performance_tracker['intermed_fusion'][fusion_name]['test_hadm_ids']
    ids_with_labels = list(label_df['hadm_id'].values)
    prediction_dict = create_prediction_dict(ecg_preds, topics_preds, fusion_preds, ecg_hadm_ids, topic_hadm_ids, fused_hadm_ids, ids_with_labels) # dict of tuples, where key is hadm_id, val is tuple of prediction
    # get average prediction score across each modality
    average_prediction = {}
    for hadm_id, preds in prediction_dict.items():
        average_prediction[hadm_id] = np.mean([pred for pred in preds if pred is not None])
    # get correlation coefficient between each pair of modalities
    """
    ecg_topic_corr = np.corrcoef(ecg_preds, topics_preds)[0, 1]
    ecg_fusion_corr = np.corrcoef(ecg_preds, fusion_preds)[0, 1]
    topic_fusion_corr = np.corrcoef(topics_preds, fusion_preds)[0, 1]
    # get correlation coefficient between each pair and the average, excluding itself
    ecg_avg_corr = np.corrcoef(ecg_preds, average_prediction.values())[0, 1]
    topic_avg_corr = np.corrcoef(topics_preds, average_prediction.values())[0, 1]
    fusion_avg_corr = np.corrcoef(fusion_preds, average_prediction.values())[0, 1]
    """
    # get performance metrics for the average predictions. auc, acc, f1, precision, recall, preds, test_labels, test_hadm_ids
    preds = list(average_prediction.values())
    test_hadm_ids = list(prediction_dict.keys())
    test_labels = [label_df[label_df['hadm_id'] == hadm_id][task].values[0] for hadm_id in test_hadm_ids]
    # use test hadm ids to index into test labels
    if task == 'los':
        test_labels = [label > 7 for label in test_labels] # binarize if we're predicting LOS
    # remove nans if either test_labels or preds has them
    nan_indices = [i for i in range(len(test_labels)) if np.isnan(test_labels[i]) or np.isnan(preds[i])]
    test_labels = [label for i, label in enumerate(test_labels) if i not in nan_indices]
    preds = [pred for i, pred in enumerate(preds) if i not in nan_indices]
    auc = roc_auc_score(test_labels, preds)
    binary_preds = [pred > 0.5 for pred in preds]
    accuracy = accuracy_score(test_labels, binary_preds)
    f1 = f1_score(test_labels, binary_preds)
    precision = precision_score(test_labels, binary_preds)
    recall = recall_score(test_labels, binary_preds)

    performance_tracker['late_fusion'][fusion_name] = {
        'auc': auc,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'preds': preds,
        'test_labels': test_labels,
        'test_hadm_ids': test_hadm_ids
    }
    # print results as always
    print(f'Late fusion results for {fusion_name}:')
    print(f'AUC: {auc}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}')

    return performance_tracker

def get_intermed_name(ecg_type, topic_type):
    return f'ecg-{ecg_type}_topics-{topic_type}'

def make_tracker(modalities, all_csvs):
    tracker = {
        modality: {feature_type: {} for feature_type in all_csvs[modality]}
               for modality in modalities}
    return tracker


if __name__ == '__main__':
    BASE_DIR = '../data/'
    ecg_csvs = {'ecg': 'filtered_ecg_features.csv'}
    topic_modes = ['labs', 'labs_notes', 'labs_notes_ecg']#, 'ecg']
    modalities = ['ecg', 'topics']
    topic_csvs = {topic: get_topic_file(topic) for topic in topic_modes}

    train_hparams = {
        'normalize': True,
        'missing_method': 'zero_fill',
        'model': 'xgboost'
        #'categorical': {'normalize': False, 'missing_method': 'zero_fill'},
        #'continuous': {'normalize': True, 'missing_method': 'zero_fill'}
    }

    tasks = {
        'los': {'csv_path': 'labels_matched.csv', 'cutoff': 14, 'label_colname': 'los'},
        'mort': {'csv_path': 'labels_matched.csv', 'cutoff': 0, 'label_colname': 'mort'}
    }

    FIRST_FEAT_COL = {
        'ecg': 22,
        'topics': 3,
    }
    save_base = '../data/supervised_pipeline/'
    all_csvs = {'ecg': ecg_csvs,
                'topics': topic_csvs}
    performance_tracker = {
        task: {modality: {feature_type: {} for feature_type in all_csvs[modality]}
               for modality in modalities} for task in tasks
    }


    ## FLAGS ##
    unimodal_flag = True
    intermediate_fusion_flag = True
    late_fusion_flag = True

    for modality in modalities: # ecg, topics
        for feature_type, csvs in all_csvs[modality].items(): # categorical vs continuous for ecg, labs vs notes vs labs_notes for topics
            for task in tasks.keys():
                if not unimodal_flag:
                    break
                print(f'running on {feature_type.upper()} features '
                      f'for {task.upper()} predictions '
                      f'with {modality.upper()} features')
                df = pd.read_csv(f'{BASE_DIR}{csvs}')
                los_df = pd.read_csv(f'{BASE_DIR}{tasks[task]["csv_path"]}')
                feat_colnames = df.columns[FIRST_FEAT_COL[modality]:]
                clf, preds, test_labels, accuracy, f1, precision, recall, auc, test_hadm_ids \
                = main(df, los_df=los_df, feat_colnames=feat_colnames, **tasks[task], **train_hparams)
                performance_tracker[task][modality][feature_type] = {
                    'auc': auc,
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'preds': preds,
                    'test_labels': test_labels,
                    'test_hadm_ids': list(test_hadm_ids),
                    'importance_scores': clf.feature_importances_
                }
                # plot importance and save pngs
                plot_importance(clf, list(feat_colnames.values), orientation='horizontal', model_name=train_hparams['model'])
                #plt.title(f'{task.upper()} {modality.upper()} {feature_type.upper()}')
                plt.savefig(f'{save_base}/{task.upper()}_{modality.upper()}_{feature_type.upper()}.png', bbox_inches='tight')
                plt.close()
    # get task_specific performance tracker
    if unimodal_flag:
        for task in tasks.keys():
            with open(f'{save_base}/{task}_performance_tracker.pkl', 'wb') as f:
                pickle.dump(performance_tracker[task], f)
        del performance_tracker


    intermed_performance_tracker = make_tracker(modalities, all_csvs)
    for task in tasks.keys():
        if not intermediate_fusion_flag:
            break
        intermed_performance_tracker['intermed_fusion'] = {}
        permutations = get_dict_permutations(intermed_performance_tracker['ecg'], intermed_performance_tracker['topics'])
        for ecg_type, topic_type in permutations: # perform intermediate fusion by loading both csvs and combining by hadm_id
            ecg_csv = all_csvs['ecg'][ecg_type]
            topics_csv = all_csvs['topics'][topic_type]

            # align predictions by hadm_ids
            ecg_df = pd.read_csv(f'{BASE_DIR}{ecg_csv}')
            topics_df = pd.read_csv(f'{BASE_DIR}{topics_csv}')
            ecg_feature_cols = ecg_df.columns[FIRST_FEAT_COL['ecg']:]
            topics_feature_cols = topics_df.columns[FIRST_FEAT_COL['topics']:]
            all_feature_cols = ecg_feature_cols.append(topics_feature_cols)
            los_df = pd.read_csv(f'{BASE_DIR}{tasks[task]["csv_path"]}')
            combined_df = pd.merge(ecg_df, topics_df, on=['split', 'hadm_id'])
            print(f'task: {task.upper()}. running on {ecg_type.upper()} and {topic_type.upper()} features...')
            clf, preds, test_labels, accuracy, f1, precision, recall, auc, test_hadm_ids \
                = main(combined_df, los_df=los_df, feat_colnames=all_feature_cols,
                       **tasks[task], **train_hparams)
            run_name = get_intermed_name(ecg_type, topic_type)
            intermed_performance_tracker['intermed_fusion'][run_name] = {
                'auc': auc,
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'preds': preds,
                'test_labels': test_labels,
                'test_hadm_ids': list(test_hadm_ids)
            }
            # plot importance and save pngs
            plot_importance(clf, list(all_feature_cols.values), orientation='horizontal',
                            model_name=train_hparams['model'])
            #plt.title(f'{task.upper()} {run_name}')
            plt.savefig(f'{save_base}/{task.upper()}_{run_name}.png')
            plt.close()
            # pickle performance tracker
            with open(f'{save_base}/{task}_intermed_fusion_performance_tracker.pkl', 'wb') as f:
                pickle.dump(intermed_performance_tracker, f)
        # get their late fusion consensus performance from 'preds'
    if not late_fusion_flag:  # we need to load unimodal and intermediate fusion performance trackers
        tasks = {}
    for task in tasks.keys():
        late_fusion_tracker = {}
        with open(f'{save_base}/{task}_performance_tracker.pkl', 'rb') as f:
            performance_tracker = pickle.load(f)
        if not intermediate_fusion_flag:  # load intermediate fusion performance tracker
            with open(f'{save_base}/{task}_intermed_fusion_performance_tracker.pkl', 'rb') as f:
                intermed_performance_tracker = pickle.load(f)
        performance_tracker = {**intermed_performance_tracker, **performance_tracker,
                               'late_fusion': {}}  # the earlier one gets matching keys overwritten
        permutations = get_dict_permutations(intermed_performance_tracker['ecg'], intermed_performance_tracker['topics'])
        los_df = pd.read_csv(f'{BASE_DIR}{tasks[task]["csv_path"]}')
        for ecg_type, topic_type in permutations:
            # combine into one tracker since they have the same inner key but different outer keys
            print(f'running late fusion on {ecg_type.upper()} and {topic_type.upper()} features...')
            # load unimodal and intermediate performance tracker
            performance_tracker = late_fusion(performance_tracker, ecg_type, topic_type, los_df, task)
            # pickle performance tracker
        with open(f'{save_base}/{task}_late_fusion_performance_tracker.pkl', 'wb') as f:
            pickle.dump(performance_tracker, f)









