import numpy as np
import torch

# ============== loc_metric ================= #

def locMetric(preds, labels):
    y_t = labels.cpu().detach().numpy()
    y_p = preds.cpu().detach().numpy()
    tp = [0 for i in range(50)]
    pre_damge = [0 for i in range(50)]
    true_damage = [0 for i in range(50)]
    true_accu = 0

    for i in range(y_p.shape[0]):
        index_t = np.where(y_t[i] > 0)[0]
        index_p = np.where(y_p[i] > 10)[0]

        if (index_t.shape == index_p.shape):
            if (index_t == index_p).all():
                true_accu += 1
        for ind in index_p:
            pre_damge[ind] += 1
            if ind in index_t:
                tp[ind] += 1

        for ind in index_t:
            true_damage[ind] += 1

    if sum(pre_damge) == 0 or sum(true_damage) == 0 or sum(tp) == 0:
        return 0, 0, 0, 0
    all_pre = sum(tp) / sum(pre_damge)
    all_recall = sum(tp) / sum(true_damage)
    all_F1 = 2 * all_pre * all_recall / (all_pre + all_recall)
    accuracy = true_accu / y_t.shape[0]
    return all_pre,all_recall,all_F1,accuracy
