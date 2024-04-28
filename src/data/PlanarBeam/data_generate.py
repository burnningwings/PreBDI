import warnings

warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math
from tqdm import tqdm

pd.options.display.max_columns = 100

import os

all_files = os.listdir('/data1/home/liujun/data/PlanarBeam/PlanarBeam_multi_Damage/')
# 50,250,500,750,950,1050,1250,1500,1750,1950
# 46,246,496,746,946,1046,1246,1496,1746,1946
# 41,241,491,741,941,1041,1241,1491,1741,1941
time_len = 128
train_sample = 35
eval_sample = 5
test_sample = 5

index = [50, 250, 500, 750, 950, 1050, 1250, 1500, 1750, 1950, 41, 241, 491, 741, 941, 1041, 1241, 1491, 1741, 1941]
index = [i - 1 for i in index]
cols = ['s' + str(i) for i in range(1, 21)]
dataT = pd.read_csv('/data1/home/liujun/data/PlanarBeam/temperature.csv', header=0,
                    names=['Var1'])
dataT['Var1'] = dataT['Var1'] / 15

noneDamage = pd.read_csv("/data1/home/liujun/data/PlanarBeam/Planar_Beam_Single_Damage/Planar_Beam_DetalT1/Planar_Beam_Single_DetalT1_S0/Planar_Beam_Single_DetalT1_P0_L0_S0.csv", header=0,names=['Var1'])
noneDamage = noneDamage.T
noneDamage = noneDamage[index]
noneDamage.columns = cols
init = noneDamage[0:1].values[0].tolist()
noneDamage = pd.concat([noneDamage, dataT], axis=1)
for i in tqdm(range(1, 21)):
    noneDamage['s' + str(i)] = dataT['Var1'] * init[i - 1]
noneDamage = noneDamage[1:]
noneDamage['loc1'] = 0
noneDamage['quant1'] = 0
noneDamage['loc2'] = 0
noneDamage['quant2'] = 0
noneDamage['loc3'] = 0
noneDamage['quant3'] = 0
# noneDamage['tempo'] = dataT['Var1'] * 15 * 10 - 105
noneDamage['tempo'] = dataT['Var1'] * 15
noneDamage['tempo'] = noneDamage['tempo']
time_in_week = [int(i % 28 + 1) for i in range(len(noneDamage))]
noneDamage['time_in_week'] = time_in_week

# data = noneDamage[:time_len * train_sample]
# eval_data = noneDamage[time_len * train_sample:time_len * (train_sample + eval_sample)]
# test_data = noneDamage[time_len * (train_sample + eval_sample):time_len * (train_sample + eval_sample + test_sample)]

file_path = "/data1/home/liujun/data/PlanarBeam/raw_data"
# data.to_csv(file_path + "/Bridge_train_single.csv", index=None)
# eval_data.to_csv(file_path + "/Bridge_eval_single.csv", index=None)
# test_data.to_csv(file_path + "/Bridge_test_single.csv", index=None)
# data.to_csv(file_path + "/Bridge_train_double.csv", index=None)
# eval_data.to_csv(file_path + "/Bridge_eval_double.csv", index=None)
# test_data.to_csv(file_path + "/Bridge_test_double.csv", index=None)
# data.to_csv(file_path + "/Bridge_train_triple.csv", index=None)
# eval_data.to_csv(file_path + "/Bridge_eval_triple.csv", index=None)
# test_data.to_csv(file_path + "/Bridge_test_triple.csv", index=None)
all_data = noneDamage[:time_len * (train_sample + eval_sample + test_sample)]
all_data.to_csv(file_path + "/Bridge_all_single.csv", index=None)
all_data.to_csv(file_path + "/Bridge_all_double.csv", index=None)
all_data.to_csv(file_path + "/Bridge_all_triple.csv", index=None)

# fileCount = 3000

import math

for fileName in tqdm(all_files):

    f = fileName.split('.')[0].split('-')
    loc1 = loc2 = loc3 = 0
    quant1 = quant2 = quant3 = 0
    if len(f) == 1:
        loc1 = int(f[0].split('+')[0])
        quant1 = int(f[0].split('+')[1])
    elif len(f) == 2:
        loc1 = int(f[0].split('+')[0])
        quant1 = int(f[0].split('+')[1])
        loc2 = int(f[1].split('+')[0])
        quant2 = int(f[1].split('+')[1])
    else:
        loc1 = int(f[0].split('+')[0])
        quant1 = int(f[0].split('+')[1])
        loc2 = int(f[1].split('+')[0])
        quant2 = int(f[1].split('+')[1])
        loc3 = int(f[2].split('+')[0])
        quant3 = int(f[2].split('+')[1])

    df = pd.read_csv("/data1/home/liujun/data/PlanarBeam/PlanarBeam_multi_Damage/" + fileName,
                     header=0, names=['Var1'])
    df = df.T
    df = df[index]
    df.columns = cols
    init = df[0:1].values[0].tolist()
    df = pd.concat([df, dataT], axis=1)
    for c in range(1, 21):
        df['s' + str(c)] = dataT['Var1'] * init[c - 1]
    df = df[1:]
    df['loc1'] = loc1
    df['quant1'] = quant1
    df['loc2'] = loc2
    df['quant2'] = quant2
    df['loc3'] = loc3
    df['quant3'] = quant3
    # df['tempo'] = dataT['Var1'] * 15 * 10 - 105
    df['tempo'] = dataT['Var1'] * 15
    df['tempo'] = df['tempo']
    time_in_week = [int(i % 28 + 1) for i in range(len(df))]
    df['time_in_week'] = time_in_week

    tmp_all = df[:time_len * (train_sample + eval_sample + test_sample)].reset_index(drop=True)

    # 只用单场景
    # if loc2 == 0 and loc3 == 0:
    #     tmp_all.to_csv(file_path + "/Bridge_all" + ".csv", index=None, mode='a', header=False)

    # tmp_all.to_csv(file_path + "/Bridge_all" + ".csv", index=None, mode='a', header=False)

    if loc2 != 0 and loc3 != 0:
        tmp_all.to_csv(file_path + "/Bridge_all_triple" + ".csv", index=None, mode='a', header=False)
    elif loc2 != 0:
        tmp_all.to_csv(file_path + "/Bridge_all_double" + ".csv", index=None, mode='a', header=False)
    else:
        tmp_all.to_csv(file_path + "/Bridge_all_single" + ".csv", index=None, mode='a', header=False)



    # tmp_train = df[:time_len * train_sample].reset_index(drop=True)
    # tmp_eval = df[time_len * train_sample: time_len * (train_sample + eval_sample)].reset_index(drop=True)
    # tmp_test = df[time_len * (train_sample + eval_sample):time_len * (train_sample + eval_sample + test_sample)].reset_index(drop=True)
    #
    # if loc2 != 0 and loc3 != 0:
    #     tmp_train.to_csv(file_path + "/Bridge_train_triple" + ".csv", index=None, mode='a', header=False)
    #     tmp_eval.to_csv(file_path + "/Bridge_eval_triple" + ".csv", index=None, mode='a', header=False)
    #     tmp_test.to_csv(file_path + "/Bridge_test_triple" + ".csv", index=None, mode='a', header=False)
    # elif loc2 != 0:
    #     tmp_train.to_csv(file_path + "/Bridge_train_double" + ".csv", index=None, mode='a', header=False)
    #     tmp_eval.to_csv(file_path + "/Bridge_eval_double" + ".csv", index=None, mode='a', header=False)
    #     tmp_test.to_csv(file_path + "/Bridge_test_double" + ".csv", index=None, mode='a', header=False)
    # else:
    #     tmp_train.to_csv(file_path + "/Bridge_train_single" + ".csv", index=None, mode='a', header=False)
    #     tmp_eval.to_csv(file_path + "/Bridge_eval_single" + ".csv", index=None, mode='a', header=False)
    #     tmp_test.to_csv(file_path + "/Bridge_test_single" + ".csv", index=None, mode='a', header=False)


    # tmp_train.to_csv(file_path + "/Bridge_train" + ".csv", index=None, mode='a', header=False)
    # tmp_eval.to_csv(file_path + "/Bridge_eval" + ".csv", index=None, mode='a', header=False)
    # tmp_test.to_csv(file_path + "/Bridge_test" + ".csv", index=None, mode='a', header=False)