import numpy as np
from math import ceil
from scipy.stats import norm

from TaPR import compute_precision_recall
from data_loader import _count_anomaly_segments

n_thresholds = 1000


def _simulate_thresholds(rec_errors, n, verbose):
    # maximum value of the anomaly score for all time steps in the test data
    thresholds, step_size = [], abs(np.max(rec_errors) - np.min(rec_errors)) / n
    th = np.min(rec_errors)

    if verbose:
        print(f'Threshold Range: ({np.max(rec_errors)}, {np.min(rec_errors)}) with Step Size: {step_size}')
    for i in range(n):
        thresholds.append(float(th))
        th = th + step_size

    return thresholds


def _flatten_anomaly_scores(values, stride, flatten=False):
    flat_seq = []
    if flatten:
        for i, x in enumerate(values):
            if i == len(values) - 1:
                flat_seq = flat_seq + list(np.ravel(x).astype(float))
            else:
                flat_seq = flat_seq + list(np.ravel(x[:stride]).astype(float))
    else:
        flat_seq = list(np.ravel(values).astype(float))

    return flat_seq


def compute_anomaly_scores(x, rec_x, scoring='square', x_val=None, rec_val=None):
    # average anomaly scores from different sensors/channels/metrics/variables (in case of multivariate time series)
    if scoring == 'absolute':
        return np.mean(np.abs(x - rec_x), axis=-1)
    elif scoring == 'square':
        return np.mean(np.square(x - rec_x), axis=-1)
    elif scoring == 'normal':
        if x_val is not None and rec_val is not None:
            val_rec_err = x_val - rec_val
            test_rec_err = x - rec_x
            mu, std = norm.fit(val_rec_err)
            return (test_rec_err - mu).T * std ** -1 * (test_rec_err - mu)


def compute_metrics(anomaly_scores, labels, label_segments=None, n=n_thresholds, delta=0.01, alpha=0.5, theta=0.5, stride=1, verbose=False):
    if label_segments is None:
        label_segments = []
    thresholds = _simulate_thresholds(anomaly_scores, n, verbose)
    correct_count, correct_ratio = [], []
    precision, recall, f1 = [], [], []

    flat_seq = _flatten_anomaly_scores(anomaly_scores, stride, flatten=len(anomaly_scores.shape) == 2)
    print('here1', len(thresholds))
    for th in thresholds:
        pred_anomalies = np.zeros(len(flat_seq)).astype(int)  # default with no anomaly
        pred_anomalies[np.where(np.array(flat_seq) > th)[0]] = 1  # assign 1 if scores > threshold
        _, pred_segments = _count_anomaly_segments(pred_anomalies)

        if len(labels) != len(pred_anomalies):
            print(f'evaluating with unmatch shape: Labels: {len(labels)} vs. Preds: {len(pred_anomalies)}')
            labels = labels[-len(pred_anomalies):]  # ref. OmniAnomaly
            print(f'evaluating with unmatch shape: Labels: {len(labels)} vs. Preds: {len(pred_anomalies)}')

        anomaly_lengths = []
        for seg in label_segments:
            anomaly_lengths.append(len(seg))
        TaD = 0 if len(anomaly_lengths) == 0 else np.ceil(np.mean(anomaly_lengths) * delta).astype(int)

        TaP, TaR = compute_precision_recall(pred_anomalies, labels, theta=theta, delta=TaD, alpha=alpha, verbose=verbose)
        count, ratio = compute_accuracy(pred_segments, label_segments, delta)

        precision.append(float(TaP))
        recall.append(float(TaR))
        f1.append(float(2 * (TaP * TaR) / (TaP + TaR + 1e-7)))
        correct_count.append(int(count))
        correct_ratio.append(float(ratio))

    return {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1': np.max(f1),
        'count': correct_count,
        'ratio': correct_ratio,
        'thresholds': thresholds,
        'anomaly_scores': flat_seq
    }


def visualization(anomaly_scores, x_test, labels):
    # TODO: visualize original data + label and anomaly_scores side-by-side

    return


def compute_accuracy(pred_segments, anomaly_segments, delta):
    correct = 0
    for seg in anomaly_segments:
        L = seg[-1] - seg[0]  # length of anomaly
        d = ceil(L * delta)
        for pred in pred_segments:
            P = pred[len(pred) // 2]  # center location as an integer

            if min([seg[0] - L, seg[0] - d]) < P < max([seg[-1] + L, seg[-1] + d]):
                correct = correct + 1
                break

    return correct, correct / (len(anomaly_segments) + 1e-7)
