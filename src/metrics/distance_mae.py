import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
import math

# ============== distance_MAE ================= #
def distanceMae(preds, labels, tod=5):
    y_t = labels.cpu().detach().numpy()
    y_p = preds.cpu().detach().numpy()
    tolerant_d = tod
    print('tolerant_d  = ', tolerant_d)
    n = y_p.shape[0]
    count = 0

    result = 0
    for i in range(n):
        index_t = np.where(y_t[i] > 0)[0]
        count += len(index_t)
        index_p = np.where(y_p[i] > 10)[0]
        if len(index_t) == 0:
            result += mean_absolute_error(y_t[i], y_p[i])
        else:
            r = 0
            counted_index_t = []
            temp_t = 0
            for p in index_p:
                d = 9999999
                for t in index_t:
                    d_temp = abs(p - t)
                    if d > d_temp:
                        d = d_temp
                        temp_t = t

                counted_index_t.append(temp_t)
                if d < tolerant_d:
                    if d < 1:
                        r += abs(y_t[i][temp_t] - y_p[i][p])
                    else:
                        r += d * abs(y_t[i][temp_t] - y_p[i][p])
                else:
                    if y_t[i][temp_t] == y_p[i][p]:
                        r += tolerant_d * y_t[i][temp_t]
                    else:
                        # r += tolerant_d * abs(y_t[i][temp_t] - y_p[i][p])
                        r += tolerant_d * y_p[i][p]

            not_counted_index_t = list(set(index_t).difference(set(counted_index_t)))
            if len(not_counted_index_t) > 0:
                for t in not_counted_index_t:
                    r += tolerant_d * y_t[i][temp_t]
            result += r

    return result / count