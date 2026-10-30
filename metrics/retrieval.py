"""DEC/DAC 3D object retrieval evaluation metrics.

This is a syntax-cleaned copy of the metric implementation supplied by the
project group. Its metric definitions are intentionally preserved for strict
comparison with DEC/DAC results.
"""

from __future__ import annotations

import numpy as np
import scipy.spatial


def eval_all_metric(
    query_fts,
    target_fts,
    query_lbls,
    target_lbls,
    metric="cosine",
):
    """Evaluate and print all DEC/DAC retrieval metrics.

    The ``metric`` argument is retained for API compatibility. The original
    implementation always uses cosine distance.
    """
    query_fts, target_fts, query_lbls, target_lbls = ex_label(
        query_fts, target_fts, query_lbls, target_lbls
    )
    dist_mat = scipy.spatial.distance.cdist(query_fts, target_fts, "cosine")
    s_map = map_score(dist_mat, query_lbls, target_lbls)
    s_ndcg = ndcg_score(dist_mat, query_lbls, target_lbls)
    s_anmrr = anmrr_score(dist_mat, query_lbls, target_lbls)
    s_recall = recall_score(dist_mat, query_lbls, target_lbls)
    s_pr = pr(dist_mat, query_lbls, target_lbls)

    print(f"{'mAP':>10s}|{'NDCG@100':>10s}|{'ANMRR':>10s}|{'Recall@100':>12s}")
    print(f"{s_map:10.5f}|{s_ndcg:10.5f}|{s_anmrr:10.5f}|{s_recall:10.5f}")
    print(f"pr curve: \n{s_pr}")

    # Returning the values does not alter the original printed interface and
    # makes it possible to save exact experiment results programmatically.
    return {
        "mAP": float(s_map),
        "NDCG@100": float(s_ndcg),
        "ANMRR": float(s_anmrr),
        "Recall@100": float(s_recall),
        "pr_curve": s_pr,
    }


def acc_score(y_true, y_pred, average="micro"):
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if average == "micro":
        return np.mean(y_true == y_pred)
    if average == "macro":
        cls_acc = []
        for cls_idx in np.unique(y_true):
            cls_acc.append(np.mean(y_pred[y_true == cls_idx] == cls_idx))
        return np.mean(np.array(cls_acc))
    raise NotImplementedError


def pr(dist_mat, lbl_a, lbl_b, top_k=1e9):
    n_a, n_b = dist_mat.shape
    top_k = int(min(top_k, n_b))
    s_idx = dist_mat.argsort()
    ans = []
    for i in range(n_a):
        cur_pr = [0] * 11
        order = s_idx[i][:top_k]
        p_list, r_list = [], []
        truth = lbl_a[i] == lbl_b[order]
        r_seen, r_max = 0, truth.sum()
        for j in range(top_k):
            if truth[j]:
                r_seen += 1
                r_list.append(r_seen / r_max)
                p_list.append(r_seen / (j + 1))
        if r_seen != 0:
            for ii in range(len(p_list)):
                p_list[ii] = max(p_list[ii:])
            r_list, p_list = np.array(r_list), np.array(p_list)
            for idx, threshold in enumerate(np.arange(0.0, 1.1, 0.1)):
                if np.sum(r_list >= threshold) != 0:
                    cur_pr[idx] = np.max(p_list[r_list >= threshold])
        ans.append(cur_pr)
    return np.array(ans).mean(0).tolist()


def map_score(dist_mat, lbl_a, lbl_b, top_k=1e9):
    n_a, n_b = dist_mat.shape
    top_k = int(min(top_k, n_b))
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        p_list = []
        relevant_seen = 0
        for j in range(top_k):
            if lbl_a[i] == lbl_b[order[j]]:
                relevant_seen += 1
                p_list.append(relevant_seen / (j + 1))
        if relevant_seen > 0:
            for ii in range(len(p_list)):
                p_list[ii] = max(p_list[ii:])
            res.append(np.array(p_list).mean())
        else:
            res.append(0)
    return np.mean(res)


def recall_score(dis_mat, lbl_a, lbl_b, top_k=100):
    n_a, n_b = dis_mat.shape
    top_k = min(top_k, n_b)
    s_idx = dis_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        relevant_seen = 0
        for j in range(top_k):
            if lbl_a[i] == lbl_b[order[j]]:
                relevant_seen += 1
        # This denominator intentionally matches the supplied DEC/DAC code.
        res.append(relevant_seen / (lbl_a == lbl_a[i]).sum())
    return np.mean(res)


def nn_score(dist_mat, lbl_a, lbl_b):
    n_a, _ = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        res.append(1 if lbl_a[i] == lbl_b[order[0]] else 0)
    return np.mean(res)


def ndcg_score(dist_mat, lbl_a, lbl_b, k=100):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, n_b + 2)))
        dcg = np.cumsum(
            [
                1.0 / np.log2(idx + 2) if lbl_a[i] == lbl_b[item] else 0.0
                for idx, item in enumerate(order)
            ]
        )
        ndcg = (dcg / idcg)[k - 1]
        res.append(ndcg)
    return np.mean(res)


def anmrr_score(dist_mat, lbl_a, lbl_b):
    n_a, _ = dist_mat.shape
    lbl_a, lbl_b = np.array(lbl_a), np.array(lbl_b)
    ng_values = np.array([(lbl_a[i] == lbl_b).sum() for i in range(lbl_a.shape[0])])
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        current_ng = ng_values[i]
        k_value = min(4 * current_ng, 2 * ng_values.max())
        order = s_idx[i]
        arr = np.sum(
            [
                (idx + 1) / current_ng
                if lbl_a[i] == lbl_b[order[idx]]
                else (k_value + 1) / current_ng
                for idx in range(current_ng)
            ]
        )
        mrr = arr - 0.5 * current_ng - 0.5
        nmrr = mrr / (k_value - 0.5 * current_ng + 0.5)
        res.append(nmrr)
    return np.mean(res)


def ex_label(query_fts, target_fts, query_lbls, target_lbls):
    """Preserve the supplied DEC/DAC class-filtering hook (currently no drops)."""
    query_fts = np.asarray(query_fts)
    target_fts = np.asarray(target_fts)
    query_lbls = np.asarray(query_lbls)
    target_lbls = np.asarray(target_lbls)
    drop_labels = []
    query_mask = np.array([label not in drop_labels for label in query_lbls])
    target_mask = np.array([label not in drop_labels for label in target_lbls])
    return (
        query_fts[query_mask],
        target_fts[target_mask],
        query_lbls[query_mask],
        target_lbls[target_mask],
    )
