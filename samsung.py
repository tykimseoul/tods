import os
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from evaluator import compute_metrics
# from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
# from tods.sk_interface.detection_algorithm.MatrixProfile_skinterface import MatrixProfileSKI
from tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import AutoEncoderSKI
from tods.sk_interface.detection_algorithm.VariationalAutoEncoder_skinterface import VariationalAutoEncoderSKI
from tods.sk_interface.detection_algorithm.HBOS_skinterface import HBOSSKI
from tods.sk_interface.detection_algorithm.So_Gaal_skinterface import So_GaalSKI
from tods.sk_interface.detection_algorithm.LODA_skinterface import LODASKI

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics

methods = {
    # 'TelemanomSKI': TelemanomSKI(l_s=10, n_predictions=10, layers=[128, 128], epochs=20),
    # 'MatrixProfileSKI': MatrixProfileSKI(),
    'AutoEncoderSKI': AutoEncoderSKI(hidden_neurons=[32, 16, 8, 16, 32]),
    # 'AutoEncoderSKI': "",
    # 'So_GaalSKI_05': So_GaalSKI(contamination=0.05),
    # 'So_GaalSKI_1': So_GaalSKI(contamination=0.1),
    # 'So_GaalSKI_2': So_GaalSKI(contamination=0.2),
    # 'So_GaalSKI_3': So_GaalSKI(contamination=0.3),
    # 'LODASKI_20': LODASKI(n_bins=20),
    # 'LODASKI_15': LODASKI(n_bins=15), *
    # 'LODASKI_10': LODASKI(n_bins=10),
    # 'LODASKI_5': LODASKI(n_bins=5),
    # 'VariationalAutoEncoderSKI': VariationalAutoEncoderSKI(encoder_neurons=[128, 64, 32, 8], decoder_neurons=[8, 32, 64, 128], epochs=30),
    # 'VariationalAutoEncoderSKI': "",
    # 'VariationalAutoEncoderSKI_16': VariationalAutoEncoderSKI(encoder_neurons=[8, 16, 8], decoder_neurons=[16, 16, 16], epochs=30),
    # 'HBOSSKI_20': HBOSSKI(n_bins=20),
    # 'HBOSSKI_15': HBOSSKI(n_bins=15),
    # 'HBOSSKI_10': HBOSSKI(n_bins=10), *
    # 'HBOSSKI_5': HBOSSKI(n_bins=5),
}


def _create_sequences(values, seq_length, stride, historical):
    seq = []
    if historical:
        for i in range(seq_length, len(values) + 1, stride):
            seq.append(values[i - seq_length:i])
    else:
        for i in range(0, len(values) - seq_length + 1, stride):
            seq.append(values[i: i + seq_length])

    print(len(seq))
    return np.stack(seq)


def _count_anomaly_segments(values):
    values = np.where(values == 1)[0]
    anomaly_segments = []

    for k, g in groupby(enumerate(values), lambda ix: ix[0] - ix[1]):
        anomaly_segments.append(list(map(itemgetter(1), g)))

    return len(anomaly_segments), anomaly_segments


def load_samsung_seg(seq_length, stride, historical=False):
    # source: SAMSUNG
    data_path = './datasets/samsung'
    datasets = sorted([f for f in os.listdir(f'{data_path}/train') if os.path.isfile(os.path.join(f'{data_path}/train', f))])

    x_train, x_test, y_test = [], [], []
    y_segment_test = []

    datasets = datasets[:1]
    for data in tqdm(datasets):
        print(data)
        train_df = np.array(pd.read_csv(f'{data_path}/train/{data}'))
        print(train_df.shape)
        train_df = train_df[:, 1:-1].astype(float)

        test_df = np.array(pd.read_csv(f'{data_path}/test/{data}'))
        labels = test_df[:, -1].astype(int)
        test_df = test_df[:, 1:-1].astype(float)
        print(test_df.shape)

        scaler = MinMaxScaler()
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)

        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        else:
            x_train.append(train_df)
            x_test.append(test_df)

        y_segment_test.append(_count_anomaly_segments(labels)[1])
        y_test.append(labels)

    return {'x_train': x_train, 'x_test': x_test, 'y_test': y_test, 'y_segment_test': y_segment_test}


samsung_data = load_samsung_seg(1, 1)
# samsung_data = np.append(samsung_data['x_train'], [samsung_data['x_test']])
# print('data', samsung_data.shape)
# data = np.loadtxt("./examples/sk_examples/500_UCR_Anomaly_robotDOG1_10000_19280_19360.txt")
# print("shape:", data.shape)
# print("datatype of data:",data.dtype)
# print("First 5 rows:\n", data[:5])

# X_train = np.expand_dims(data[:10000], axis=1)
# X_test = np.expand_dims(data[10000:], axis=1)
# print(X_train.shape, X_test.shape)

x_train = np.array(samsung_data['x_train'])
x_test = np.array(samsung_data['x_test'])
y_test = np.array(samsung_data['y_test'])
# y_test_seg = np.array(samsung_data['y_segment_test'])
y_test_seg = samsung_data['y_segment_test']
print(y_test_seg)
print(x_train.shape, x_test.shape, y_test.shape)
x_train = np.squeeze(x_train, axis=0)
x_train = np.squeeze(x_train, axis=1)
x_test = np.squeeze(x_test, axis=0)
x_test = np.squeeze(x_test, axis=1)
y_test = np.squeeze(y_test, axis=0)
print(x_train.shape, x_test.shape, y_test.shape)

# print("First 5 rows train:\n", x_train[:5])
# print("First 5 rows test:\n", x_test[:5])

for idx, method_name in enumerate(methods.keys()):
    plt.figure(figsize=(42, 6 * 2), dpi=100)
    print(f'Running {method_name}')
    # try:
    transformer = methods[method_name]
    transformer.fit(x_train)
    prediction_labels_train = transformer.predict(x_train)

    decision_score_train = transformer.primitives[0]._clf.decision_function(x_train)
    threshold_train = transformer.primitives[0]._clf.threshold_
    labels_train = transformer.primitives[0]._clf.labels_
    print('here', decision_score_train.shape, threshold_train, labels_train.shape)

    prediction_labels_test = transformer.predict(x_test)
    prediction_score = transformer.predict_score(x_test)
    y_pred = prediction_labels_test

    decision_score_test = transformer.primitives[0]._clf.decision_function(x_test)
    # threshold_test = transformer.primitives[0]._clf.threshold_
    # labels_test = transformer.primitives[0]._clf.labels_
    # print('here', decision_score_test.shape, threshold_test, labels_test.shape)


    print("Prediction Labels\n", prediction_labels_test.shape)
    print("Prediction Score\n", prediction_score.shape)

    y_test_cropped = y_test[-8000:]
    x_test_cropped = x_test[-8000:, :]
    x_train_cropped = x_train[-8000:, :]
    prediction_labels_train_cropped = prediction_labels_train[-8000:]
    prediction_labels_test_cropped = prediction_labels_test[-8000:]
    decision_score_train_cropped = decision_score_train[-8000:]
    decision_score_test_cropped = decision_score_test[-8000:]

    plt.subplot(4, 1, 1)
    plt.title(f'{method_name} (Training Set)')
    # plt.vlines((y_test_cropped == 1).nonzero(), np.amin(x_train_cropped), np.amax(x_train_cropped), color='green', linewidth=0.5, linestyles='solid', alpha=0.5, label='Ground Truth')
    plt.vlines((prediction_labels_train_cropped == 1).nonzero(), np.amin(x_train_cropped), np.amax(x_train_cropped), color='red', linewidth=0.5, linestyles='solid', alpha=0.5, label='Prediction')
    plt.plot(x_train_cropped, linewidth=0.4)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

    plt.subplot(4, 1, 2)
    plt.title(f'{method_name} Threshold: {threshold_train}')
    plt.axhline(y=threshold_train, color='r', linewidth=0.5, linestyle='-', alpha=0.5)
    plt.plot(decision_score_train_cropped, linewidth=0.4)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

    # y_true = prediction_labels_train

    print('Accuracy Score: ', accuracy_score(y_test, y_pred))

    confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)
    f1_scores = 2 * recalls * precisions / (recalls + precisions)

    print('Best threshold: ', thresholds[np.argmax(f1_scores)])
    print('Best F1-Score: ', np.max(f1_scores))

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('Precision', precision)
    print('Recall', recall)
    print('F1 score', f1)

    print('visualizing..')
    Path('./viz').mkdir(exist_ok=True)

    plt.subplot(4, 1, 3)
    plt.title(f'{method_name} (F1 = {np.max(f1_scores):.3f})')
    plt.vlines((y_test_cropped == 1).nonzero(), np.amin(x_test_cropped), np.amax(x_test_cropped), color='green', linewidth=0.5, linestyles='solid', alpha=0.5, label='Ground Truth')
    plt.vlines((prediction_labels_test_cropped == 1).nonzero(), np.amin(x_test_cropped), np.amax(x_test_cropped), color='red', linewidth=0.5, linestyles='solid', alpha=0.5, label='Prediction')
    plt.plot(x_test_cropped, linewidth=0.4)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

    plt.subplot(4, 1, 4)
    plt.title(f'{method_name}')
    # plt.axhline(y=threshold_test, color='r', linewidth=0.5, linestyle='-', alpha=0.5)
    plt.plot(decision_score_test_cropped, linewidth=0.4)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

    plt.tight_layout(pad=4)
    plt.savefig(f'./viz/{method_name}.png')
    plt.show()

    plt.figure()
    plt.plot(recalls, precisions, marker='.', label=method_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'./viz/prc_{method_name}.png')
    plt.show()

    print('visualization done')

    print('Evaluating..')
    print(decision_score_test.shape, np.squeeze(prediction_labels_test, axis=1).shape)
    # decision_score_test_squeezed = np.squeeze(decision_score_test, axis=1)
    prediction_labels_test = np.squeeze(prediction_labels_test, axis=1)
    # decision_score_test = _create_sequences(decision_score_test, 36, 1, historical=False)
    # prediction_labels_test = _create_sequences(prediction_labels_test, 36, 1, historical=False)
    # seg = _create_sequences(y_test_seg[0], 36, 1, historical=False)
    print('score squeezed', decision_score_test.shape, prediction_labels_test.shape, y_test_seg)
    met = compute_metrics(decision_score_test, prediction_labels_test, y_test_seg[0], delta=240, alpha=0.65, theta=0.2, stride=1, verbose=False)
    print(met['precision'], met['recall'], met['f1'])

# fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
# roc_auc = metrics.auc(fpr, tpr)
# plt.figure()
# plt.title('ROC')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig('roc.png')
# plt.show()
# except ValueError as e:
#     print(f'ValueError in {method_name}: {e}')
