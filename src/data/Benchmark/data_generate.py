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

all_files = os.listdir('/data1/home/liujun/data/Benchmark/')
time_len = 128
train_sample = 35
eval_sample = 5
test_sample = 5

cols = ['s' + str(i) for i in range(1, 33)]
dataT = pd.read_csv('/data1/home/liujun/data/PlanarBeam/temperature.csv', header=0,
                    names=['tempo'])
dataT['tempo'] = dataT['tempo'] / 15

file_path = "/data1/home/liujun/code/mvts_transformer/src/data/Benchmark/"


header = cols + ["tempo", "loc", "quant", "time_in_week"]
frame = pd.DataFrame(columns=header)
frame.to_csv(file_path + "/Bridge_all.csv", index=None)

# fileCount = 3000

import math

for fileName in tqdm(all_files):
    if fileName == ".uuid":
        continue;
    f = fileName.split('.')[0].split('_')
    loc1 = int(f[1]) - 1
    quant1 = int(f[3][:-1])

    df = pd.read_csv("/data1/home/liujun/data/Benchmark/" + fileName,
                     header=0, names=cols)

    init = df[0:1].values[0].tolist()
    df = pd.concat([df, dataT], axis=1)
    for c in range(1, 33):
        df['s' + str(c)] = dataT['tempo'] * init[c - 1]
    df['loc'] = loc1 - 2881
    df['quant'] = quant1
    df['tempo'] = dataT['tempo'] * 15
    time_in_week = [int(i % 28 + 1) for i in range(len(df))]
    df['time_in_week'] = time_in_week

    tmp_all = df[:time_len * (train_sample + eval_sample + test_sample)].reset_index(drop=True)

    tmp_all.to_csv(file_path + "/Bridge_all" + ".csv", index=None, mode='a', header=False)