import pickle
import warnings
import random

warnings.filterwarnings('ignore')

import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
pd.options.display.max_columns = 100
import os

def stand(x):
    scaler  = StandardScaler().fit(x)
    x_scaler = scaler.transform(x)
    return x_scaler

from sklearn.preprocessing import MinMaxScaler
def max_min(x):
    scaler  = MinMaxScaler().fit(x)
    x_scaler = scaler.transform(x)
    return x_scaler


all_files = os.listdir('/home/liujun/BHM/data/ASCE benchmark data-实验数据/FE/')
time_len = 128
train_sample = 374
eval_sample = 47
test_sample = 47

cols = ['s' + str(i) for i in range(1, 17)]

file_path = "/home/liujun/BHM/mvts_transformer/src/data/BenchmarkEXP/"
header = cols + ["case"]
frame = pd.DataFrame(columns=header)

input_len = 128
data_FE = []
label_FE = []
domain_FE = []
import math
for fileName in tqdm(all_files):
    f = fileName.split('.')[0]
    case = int(f[-1:]) - 1

    mat_data = sio.loadmat('/home/liujun/BHM/data/ASCE benchmark data-实验数据/FE/' + fileName)

    frame = pd.DataFrame(columns=header)

    number = f[-2:-1]
    frame['s1'] = pd.Series(mat_data['acc'][:, 0])
    frame['s2'] = pd.Series(mat_data['acc'][:, 1])
    frame['s3'] = pd.Series(mat_data['acc'][:, 2])
    frame['s4'] = pd.Series(mat_data['acc'][:, 3])
    frame['s5'] = pd.Series(mat_data['acc'][:, 4])
    frame['s6'] = pd.Series(mat_data['acc'][:, 5])
    frame['s7'] = pd.Series(mat_data['acc'][:, 6])
    frame['s8'] = pd.Series(mat_data['acc'][:, 7])
    frame['s9'] = pd.Series(mat_data['acc'][:, 8])
    frame['s10'] = pd.Series(mat_data['acc'][:, 9])
    frame['s11'] = pd.Series(mat_data['acc'][:, 10])
    frame['s12'] = pd.Series(mat_data['acc'][:, 11])
    frame['s13'] = pd.Series(mat_data['acc'][:, 12])
    frame['s14'] = pd.Series(mat_data['acc'][:, 13])
    frame['s15'] = pd.Series(mat_data['acc'][:, 14])
    frame['s16'] = pd.Series(mat_data['acc'][:, 15])

    # list_label = [0 for i in range(9)]
    # list_label[case] = 1

    df = frame[:][cols]
    sensor_data = df.values

    # normalized
    # for i in range(sensor_data.shape[1]):
    #     sd = stand(sensor_data[:, i].reshape(-1, 1))
    #     sensor_data[:, i] = sd.reshape(-1)
    sensor_data = max_min(sensor_data)

    last_index = input_len
    splited = []
    label_splited = []
    domain_splited = []
    last_index = time_len
    while last_index <= sensor_data.shape[0]:
        splited.append(sensor_data[last_index - input_len: last_index])
        label_splited.append(case)
        domain_splited.append(0)
        last_index += 8

    data_FE.append(splited)
    label_FE.append(label_splited)
    domain_FE.append(domain_splited)



all_files = os.listdir('/home/liujun/BHM/data/ASCE benchmark data-实验数据/files_bldg_shm_exp2/CD of UBC.experiment 2002 2/data/Ambient/')
time_len = 128
train_sample = 374
eval_sample = 47
test_sample = 47

cols = ['s' + str(i) for i in range(1, 17)]

file_path = "/home/liujun/BHM/mvts_transformer/src/data/BenchmarkEXP/"

header = cols + ["case"]
frame = pd.DataFrame(columns=header)

input_len = 128
data_real = []
label_real = []
domain_real = []
# sim_data_eval = []
# label_eval = []
# sim_data_test = []
# label_test = []
import math

train_real = []
tl = []
td = []
eval_real = []
el = []
ed = []
for fileName in tqdm(all_files):
    f = fileName.split('.')[0]
    case = int(f[-3:-1]) - 1

    mat_data = sio.loadmat("/home/liujun/BHM/data/ASCE benchmark data-实验数据/files_bldg_shm_exp2/CD of UBC.experiment 2002 2/data/Ambient/" + fileName)

    frame = pd.DataFrame(columns=header)

    number = f[-3:-1]
    frame['s1'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0,0]['DA01']))
    frame['s2'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA02']))
    frame['s3'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA03']))
    frame['s4'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA04']))
    frame['s5'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA05']))
    frame['s6'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA06']))
    frame['s7'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA07']))
    frame['s8'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA08']))
    frame['s9'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA09']))
    frame['s10'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA10']))
    frame['s11'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA11']))
    frame['s12'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA12']))
    frame['s13'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA13']))
    frame['s14'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA14']))
    frame['s15'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA15']))
    frame['s16'] = pd.Series(map(lambda x:x[0], mat_data['dasy'][0, 0]['DA16']))

    list_label = [0 for i in range(9)]
    list_label[case] = 1

    df = frame[:][cols]
    sensor_data = df.values

    # normalized
    # for i in range(sensor_data.shape[1]):
    #     sd = stand(sensor_data[:, i].reshape(-1, 1))
    #     sensor_data[:, i] = sd.reshape(-1)
    sensor_data = max_min(sensor_data)

    last_index = input_len
    splited = []
    label_splited = []
    domain_splited = []
    while last_index <= sensor_data.shape[0] and len(splited) < 7486:
        splited.append(sensor_data[last_index - input_len: last_index])
        # label_splited.append(case)
        label_splited.append(list_label)
        domain_splited.append(1)
        last_index += 16

    ttr = []
    ttl = []
    ttd = []
    ter = []
    tel = []
    ted = []
    # et_indices = random.sample(range(1, len(splited)), len(splited) // 5)
    # random.shuffle(et_indices)
    et_indices = [i for i in range(len(splited) // 5)]
    # et_indices = []
    for i in range(len(splited)):
        if i in et_indices:
            ttr.append(splited[i])
            ttl.append(label_splited[i])
            ttd.append(domain_splited[i])
        else:
            ter.append(splited[i])
            tel.append(label_splited[i])
            ted.append(domain_splited[i])
    train_real.append(ttr)
    tl.append(ttl)
    td.append(ttd)
    eval_real.append(ter)
    el.append(tel)
    ed.append(ted)

    data_real.append(splited)
    label_real.append(label_splited)
    domain_real.append(domain_splited)


# # train_data = data_FE + train_real
# # train_label = label_FE + tl
# # train_domain = domain_FE + td
#
# train_data = data_FE
# train_label = label_FE
# train_domain = domain_FE
#
# # eval_data = eval_real
# # eval_label = el
# # eval_domain = ed
#
# eval_data = data_real
# eval_label = label_real
# eval_domain = domain_real
#
# train_data = np.asarray(np.concatenate(train_data, axis=0), dtype=np.float32)
# train_label = np.asarray(np.concatenate(train_label, axis=0), dtype=np.int8)
# train_domain = np.asarray(np.concatenate(train_domain, axis=0), dtype=np.int8)
#
# # train_real_data = np.asarray(np.concatenate(train_real_data, axis=0), dtype=np.float32)
# # train_real_label = np.asarray(np.concatenate(train_real_label, axis=0), dtype=np.int8)
# # train_real_domain = np.asarray(np.concatenate(train_real_domain, axis=0), dtype=np.int8)
#
# train_data = np.expand_dims(train_data, axis=-1)
# # train_real_data = np.expand_dims(train_real_data, axis=-1)
# process_data = train_data
# # process_real_data = train_real_data
# data = {}
# data["data"] = process_data
# data["label"] = train_label
# data["domain"] = train_domain
# # data["real_data"] = process_real_data
# # data["real_label"] = train_real_label
# # data["real_domain"] = train_real_domain
# with open(file_path + "/Benchmark_domain_mma_0%pre_train.pkl", "wb") as f:
#     pickle.dump(data, f)
#
#
# eval_data = np.asarray(np.concatenate(eval_data, axis=0), dtype=np.float32)
# eval_label = np.asarray(np.concatenate(eval_label, axis=0), dtype=np.int8)
# eval_domain = np.asarray(np.concatenate(eval_domain, axis=0), dtype=np.int8)
# eval_data = np.expand_dims(eval_data, axis=-1)
# process_data = eval_data
# data = {}
# data["data"] = process_data
# data["label"] = eval_label
#
#
#
# data["domain"] = eval_domain
# with open(file_path + "/Benchmark_domain_mma_0%pre_eval.pkl", "wb") as f:
#     pickle.dump(data, f)
# pass


# train_real = []
# tl = []
# td = []
# eval_real = []
# el = []
# ed = []
#
# et_indices = random.sample(range(1, len(data_real)), len(data_real) // 2)
# random.shuffle(et_indices)
# for i in range(len(data_real)):
#     if i in et_indices:
#         train_real.append(data_real[i])
#         tl.append(label_real[i])
#         td.append(domain_real[i])
#     else:
#         eval_real.append(data_real[i])
#         el.append(label_real[i])
#         ed.append(domain_real[i])

train_data = np.asarray(np.concatenate(train_real, axis=0), dtype=np.float32)
train_label = np.asarray(np.concatenate(tl, axis=0), dtype=np.float32)
train_data = np.expand_dims(train_data, axis=-1)
process_data = train_data
data = {}
data["data"] = process_data
data["label"] = train_label
with open(file_path + "/Benchmark_stride16exp20%_train.pkl", "wb") as f:
    pickle.dump(data, f)


eval_data = np.asarray(np.concatenate(eval_real, axis=0), dtype=np.float32)
eval_label = np.asarray(np.concatenate(el, axis=0), dtype=np.float32)
eval_data = np.expand_dims(eval_data, axis=-1)
process_data = eval_data
data = {}
data["data"] = process_data
data["label"] = eval_label
with open(file_path + "/Benchmark_stride16exp20%_eval.pkl", "wb") as f:
    pickle.dump(data, f)
pass