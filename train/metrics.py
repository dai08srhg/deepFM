import numpy as np
from sklearn.metrics import log_loss, roc_curve, auc


def pr_auc(true_labels: np.ndarray, pred_probs: np.ndarray):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    pr_auc = auc(fpr, tpr)
    return pr_auc


def logloss(true_labels: np.ndarray, pred_probs: np.ndarray):
    loss = log_loss(true_labels, pred_probs)
    return loss


def normalized_entropy(true_labels: np.ndarray, pred_probs: np.ndarray):
    p = np.mean(pred_probs)
    logloss = log_loss(true_labels, pred_probs)
    deno = - (p * np.log(p) + (1 - p) * np.log(1 - p))
    return logloss / deno