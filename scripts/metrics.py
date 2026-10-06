"""Classification metrics in pure numpy (no scikit-learn needed).

Used by the A2 / A3 / aggregation scripts so nothing has to be pip-installed on the
server. All functions take 1-D array-likes.
  y   : integer labels (0/1)
  pred: integer predictions (0/1)     -> accuracy, mcc, sn, sp, bacc, f1
  s   : real-valued positive-class scores/probabilities -> auc, pr_auc
"""
import math
import numpy as np


def _counts(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def accuracy(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    return float((y == pred).mean()) if len(y) else float("nan")


def mcc(y, pred):
    tp, tn, fp, fn = _counts(y, pred)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def sn_sp(y, pred):
    tp, tn, fp, fn = _counts(y, pred)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sn), float(sp)


def bacc(y, pred):
    sn, sp = sn_sp(y, pred)
    return float((sn + sp) / 2)


def f1(y, pred):
    tp, tn, fp, fn = _counts(y, pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def _rankdata_avg(a):
    """Ranks with ties averaged (like scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    sa = a[order]
    i = 0
    n = len(sa)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def auc(y, s):
    """ROC-AUC via the Mann-Whitney statistic (tie-aware)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = _rankdata_avg(s)
    rank_pos = ranks[y == 1].sum()
    return float((rank_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(y, s):
    """Area under precision-recall curve (trapezoidal over sorted scores)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    P = int((y == 1).sum())
    if P == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    recall = tp / P
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    # manual trapezoidal rule (np.trapz/np.trapezoid name varies across numpy versions)
    return float(np.sum((precision[1:] + precision[:-1]) / 2.0 * np.diff(recall)))


def all_metrics(y, s, threshold=0.5):
    """Everything at once, given labels y and positive-class scores s."""
    pred = (np.asarray(s, dtype=float) >= threshold).astype(int)
    sn, sp = sn_sp(y, pred)
    return {
        "acc": accuracy(y, pred), "auc": auc(y, s), "pr_auc": pr_auc(y, s),
        "mcc": mcc(y, pred), "f1": f1(y, pred), "sn": sn, "sp": sp, "bacc": bacc(y, pred),
        "n": int(len(y)), "n_pos": int(np.asarray(y).sum()), "threshold": float(threshold),
    }
