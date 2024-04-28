import numpy as np
import torch

# ============== damage_MAE ================= #
def damageMae(preds, labels):

    y_t = labels.cpu().detach().numpy()
    y_p = preds.cpu().detach().numpy()

    index_t = np.where(y_t > 0, y_t, 0)
    index_p = np.where(y_p > 10, y_p, 0)
    count = 0
    cha = 0
    count = len(np.where(y_t > 0)[0])
    cha = np.sum(np.abs(np.subtract(index_p, index_t)))
    return cha / count
