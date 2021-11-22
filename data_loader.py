# Dataset Loader
import os
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
for each dataset
    x_train shape: (number of samples, sequence length, number of features)
    x_test shape: (number of samples, sequence length, number of features)
    y_test shape: (number of individual samples)
"""


# Generated training sequences for use in the model.
def _create_sequences(values, seq_length, stride, historical):
    seq = []
    if historical:
        for i in range(seq_length, len(values) + 1, stride):
            seq.append(values[i - seq_length:i])
    else:
        for i in range(0, len(values) - seq_length + 1, stride):
            seq.append(values[i: i + seq_length])

    return np.stack(seq)


def _count_anomaly_segments(values):
    values = np.where(values == 1)[0]
    anomaly_segments = []

    for k, g in groupby(enumerate(values), lambda ix: ix[0] - ix[1]):
        anomaly_segments.append(list(map(itemgetter(1), g)))
    return len(anomaly_segments), anomaly_segments


# 4 Univariate Datasets: Yahoo S5 A1, UCR TSAD (KDD CUP 2021), AIOPS KPI
def load_UCR(seq_length=128, stride=1, historical=False):
    # seq. length: 128:64 (random)
    # source: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
    # interval: varies
    # remark: This dataset includes: EPG, NASA (KDD), ECG, PVC, Respiration, EPG, Power Demand, Internal Bleeding etc.

    path = f'./datasets/UCR'
    f_names = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []

    for f_name in f_names:
        df = pd.read_csv(f'{path}/{f_name}', header=None, dtype=float).values
        idx = np.array(f_name.split('.')[0].split('_')[-3:]).astype(int)
        train_idx, label_start, label_end = idx[0], idx[1], idx[2] + 1
        labels = np.zeros(df.shape[0], dtype=int)
        labels[label_start:label_end] = 1

        train_df = df[:train_idx]
        test_df = df[train_idx:]
        labels = labels[train_idx:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)

        valid_idx = int(test_df.shape[0] * 0.3)
        valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]

        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
            x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        else:
            x_train.append(train_df)
            x_valid.append(valid_df)
            x_test.append(test_df)

        valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]

        y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
        y_segment_test.append(_count_anomaly_segments(test_labels)[1])

        y_valid.append(valid_labels)
        y_test.append(test_labels)

    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_KPI(seq_length=60, stride=1, historical=False):
    # seq. length: 60:30 (i.e., 1 hour)
    # source: http://iops.ai/competition_detail/?competition_id=5&flag=1
    # interval: 1 miniute (60 secs)

    path = f'./datasets/KPI'
    f_names = sorted([f for f in os.listdir(f'{path}/train') if os.path.isfile(os.path.join(f'{path}/train', f))])

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []

    for f_name in f_names:
        df = pd.read_csv(f'{path}/train/{f_name}')

        test_idx = int(df.shape[0] * 0.4)  # train 60% test 40%
        train_df = df['value'].iloc[:-test_idx].values.reshape(-1, 1)
        test_df = df['value'].iloc[-test_idx:].values.reshape(-1, 1)
        labels = df['label'].iloc[-test_idx:].values.astype(int)

        # filtering anomaly contamination
        if len(np.where(df['label'].iloc[:-test_idx].values.astype(int) == 1)[0]) == 0:

            scaler = MinMaxScaler(feature_range=(0, 1))
            train_df = scaler.fit_transform(train_df)
            test_df = scaler.transform(test_df)

            valid_idx = int(test_df.shape[0] * 0.3)
            valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]

            if seq_length > 0:
                x_train.append(_create_sequences(train_df, seq_length, stride, historical))
                x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
                x_test.append(_create_sequences(test_df, seq_length, stride, historical))
            else:
                x_train.append(train_df)
                x_valid.append(valid_df)
                x_test.append(test_df)

            valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]

            y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
            y_segment_test.append(_count_anomaly_segments(test_labels)[1])

            y_valid.append(valid_labels)
            y_test.append(test_labels)

    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


# 3 Multivariate Datasets: ASD, TODS, SWaT
def load_ASD(seq_length=100, stride=1, historical=False):
    # seq. length: 100:50 (ref. OmniAnomaly)
    # source: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
    # interval:  equally-spaced 5 minutes apart
    path = f'./datasets/ASD'
    f_names = sorted([f for f in os.listdir(f'{path}/train') if os.path.isfile(os.path.join(f'{path}/train', f))])

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []

    for f_name in f_names:
        train_df = pd.read_pickle(f'{path}/train/{f_name}')
        test_df = pd.read_pickle(f'{path}/test/{f_name}')
        labels = pd.read_pickle(f'{path}/test_label/{f_name}').astype(int)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)

        valid_idx = int(test_df.shape[0] * 0.3)
        valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]

        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
            x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        else:
            x_train.append(train_df)
            x_valid.append(valid_df)
            x_test.append(test_df)

        valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]

        y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
        y_segment_test.append(_count_anomaly_segments(test_labels)[1])

        y_valid.append(valid_labels)
        y_test.append(test_labels)

    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_TODS(seq_length=100, stride=1, historical=False):
    path = f'./datasets/TODS'

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []

    train_df = pd.read_csv(f'{path}/train.csv').drop(columns=['anomaly'])
    test_df = pd.read_csv(f'{path}/test.csv').drop(columns=['anomaly'])
    labels = pd.read_csv(f'{path}/test.csv')['anomaly'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]

    if seq_length > 0:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)

    valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]

    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(test_labels)[1])

    y_valid.append(valid_labels)
    y_test.append(test_labels)

    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_SWaT(seq_length=60, stride=30, historical=False):
    # seq. length: 600:300 (i.e., 10 minutes)
    # source: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
    # interval: 1 second

    path = f'./datasets/SWaT/downsampled'

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []

    train_df = np.load(f'{path}/train.npy')
    test_df = np.load(f'{path}/test.npy')
    labels = np.load(f'{path}/test_label.npy')

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]

    if seq_length > 0:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)

    valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]

    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(test_labels)[1])

    y_valid.append(valid_labels)
    y_test.append(test_labels)

    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}
