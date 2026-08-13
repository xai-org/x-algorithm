import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self, df, thresholds=None):
        self.df = df
        self.y_true = df["label"] == 1
        self.thresholds = (
            thresholds
            if thresholds is not None
            else np.linspace(0.0, 1.0, 200, dtype=np.float32)
        )

    def at_threshold(self, threshold):
        self.y_pred = self.df["pred"] >= threshold
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        precision = 1 if (tp + fp == 0) else tp / (tp + fp)
        recall = 1 if (tp + fn == 0) else tp / (tp + fn)
        fp_rate = 1 if (fp + tn == 0) else fp / (fp + tn)
        return [threshold, precision, recall, fp_rate]

    def at_all_thresholds(self):
        return [self.at_threshold(t) for t in self.thresholds]


def pos_ratio_at_threshold(from_threshold, to_threshold, predictions_and_labels):
    within_thresholds = predictions_and_labels[
        (predictions_and_labels["pred"] >= from_threshold)
        & (predictions_and_labels["pred"] < to_threshold)
    ]
    total = within_thresholds.size
    pos_count = within_thresholds[within_thresholds["label"] == 1].size
    res = to_threshold if total == 0 else pos_count / total
    return res


def get_interval(score, thresholds):
    return (
        max(filter(lambda i: i < score, thresholds)),
        min(filter(lambda i: i > score, thresholds)),
    )


def pos_ratio_for_score(score, predictions_and_labels, thresholds):
    (from_t, to_t) = get_interval(score, thresholds)
    return pos_ratio_at_threshold(from_t, to_t, predictions_and_labels)


def fit_isotonic_regression(predictions_and_labels, thresholds):
    X = predictions_and_labels["pred"].to_numpy()
    X = np.sort(X)
    y = [pos_ratio_for_score(x, predictions_and_labels, thresholds) for x in X]
    y = np.array(y)
    iso_reg = IsotonicRegression(
        y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
    ).fit(X, y)
    return iso_reg


def calibrate_scores(calibration_ds, thresholds):
    metrics = Metrics(calibration_ds, thresholds)
    iso_reg = fit_isotonic_regression(calibration_ds, thresholds)
    calibrated_probabilities = iso_reg.predict(metrics.thresholds)
    metrics_at_thresholds = metrics.at_all_thresholds()

    res_df = pd.DataFrame(
        metrics_at_thresholds,
        columns=[
            "thresholds_adult",
            "precision_at_thresholds_adult",
            "recall_at_thresholds_adult",
            "false_positive_rate_at_thresholds_adult",
        ],
    )
    res_df["probability_at_thresholds_adult"] = calibrated_probabilities
    return res_df


def calibrate_and_write_scores(config):
    calibration_ds = pd.read_csv(config.dataset_path, names=["pred", "label"], header=0)

    thresholds = np.linspace(0.0, 1.0, config.num_thresholds, dtype=np.float32)

    res_df = calibrate_scores(calibration_ds, thresholds)
    res_df.to_csv(config.output_dir + "/metrics_at_thresholds.csv", index=False)
