import os
import random
import sys
import shutil
import pickle
import argparse

import numpy as np
import math
from tqdm import tqdm

# TODO: remove it when basicts can be installed by pip
# sys.path.append(os.path.abspath(__file__ + "/../../../.."))
os.chdir("/data1/home/liujun/code/mvts_transformer")  # 把目录切换到当前项目，这句话是关键
import pandas as pd

def sc_normalize(x, sc_mean, sc_std):
    return (x - sc_mean) / sc_std


def add_noise(x, snr):
    snr = 10.0 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return x + np.random.randn(len(x)) * np.sqrt(npower)


def generate_noised_data(args: argparse.Namespace):
    print(os.getcwd())

    target_channel = args.target_channel
    input_len = args.input_len
    label_len = args.label_len
    node_num = args.node_num
    damage_degree = args.damage_degree
    add_time_of_week = args.tow
    add_temp = args.temp
    output_dir = args.output_dir
    data_file_path = args.data_file_path
    split_case = args.split_case
    change_step = args.change_step
    add_gwn = args.add_gwn
    gwn = args.gwn

    # read data
    train_sample = 35
    eval_sample = 5
    test_sample = 5
    train_time_len = input_len * train_sample
    val_time_len = input_len * eval_sample
    test_time_len = input_len * test_sample

    cols = ['s' + str(i) for i in range(1, 33)]
    label_cols = ["loc", "quant"]
    extend_cols = ["tempo", "time_in_week"]

    time_len = train_time_len + val_time_len + test_time_len
    data = pd.read_csv("{}_all.csv".format(data_file_path))
    sim_data_train = []
    label_train = []
    extend_train = []
    sim_data_eval = []
    label_eval = []
    extend_eval = []
    sim_data_test = []
    label_test = []
    extend_test = []

    clabel_train = []
    clabel_eval = []
    clabel_test = []

    if split_case:
        et_indices = random.sample(range(data.shape[0] // time_len), math.ceil((data.shape[0] // time_len) * 0.2))
        random.shuffle(et_indices)
        eval_indices = et_indices[:len(et_indices) // 2]
        test_indices = et_indices[len(et_indices) // 2:]

    for n in tqdm(range(data.shape[0] // time_len)):
        df = data[time_len * n: time_len * (n + 1)][cols]
        df_label = data[time_len * n: time_len * (n + 1)][label_cols]
        df_extend = data[time_len * n: time_len * (n + 1)][extend_cols]

        list_label = [0 for j in range(label_len)]
        l1 = df_label['loc'][time_len * n]
        q1 = df_label['quant'][time_len * n]
        list_label[l1] = q1

        classifier_label = [0 for j in range(node_num * damage_degree)]
        classifier_label[l1 + label_len * (int(q1 / 30) - 1)] = 1

        sensor_data = df.values
        if add_gwn:
            sensor_data = sensor_data.T
            for i in range(len(sensor_data)):
                sensor_data[i] = add_noise(sensor_data[i], gwn)
            sensor_data = sensor_data.T

        if not change_step:
            splited = np.split(sensor_data, sensor_data.shape[0] / input_len, axis=0)
            extend_splited = np.split(df_extend.values, df_extend.shape[0] / input_len, axis=0)
        else:
            splited = []
            extend_splited = []
            last_index = input_len
            while last_index <= sensor_data.shape[0]:
                splited.append(sensor_data[last_index - input_len: last_index])
                extend_splited.append(df_extend.values[last_index - input_len: last_index])
                last_index += 8

        if split_case:
            if n in eval_indices:
                for j in range(len(splited)):
                    sim_data_eval.append(splited[j])
                    label_eval.append(list_label)
                    extend_eval.append(extend_splited[j])
                    clabel_eval.append(classifier_label)
            elif n in test_indices:
                for j in range(len(splited)):
                    sim_data_test.append(splited[j])
                    label_test.append(list_label)
                    extend_test.append(extend_splited[j])
                    clabel_test.append(classifier_label)
            else:
                for j in range(len(splited)):
                    sim_data_train.append(splited[j])
                    label_train.append(list_label)
                    extend_train.append(extend_splited[j])
                    clabel_train.append(classifier_label)
        else:
            for j in range(0, (int)(len(splited) * 7 / 9)):
                sim_data_train.append(splited[j])
                label_train.append(list_label)
                extend_train.append(extend_splited[j])
                clabel_train.append(classifier_label)
            for j in range((int)(len(splited) * 7 / 9), (int)(len(splited) * 8 / 9)):
                sim_data_eval.append(splited[j])
                label_eval.append(list_label)
                extend_eval.append(extend_splited[j])
                clabel_eval.append(classifier_label)
            for j in range((int)(len(splited) * 8 / 9), len(splited)):
                sim_data_test.append(splited[j])
                label_test.append(list_label)
                extend_test.append(extend_splited[j])
                clabel_test.append(classifier_label)

    sim_data_train = np.asarray(sim_data_train, dtype=np.float32)
    extend_train = np.asarray(extend_train, dtype=np.float32)
    label_train = np.asarray(label_train, dtype=np.float32)
    sim_data_eval = np.asarray(sim_data_eval, dtype=np.float32)
    extend_eval = np.asarray(extend_eval, dtype=np.float32)
    label_eval = np.asarray(label_eval, dtype=np.float32)
    sim_data_test = np.asarray(sim_data_test, dtype=np.float32)
    extend_test = np.asarray(extend_test, dtype=np.float32)
    label_test = np.asarray(label_test, dtype=np.float32)

    clabel_train = np.asarray(clabel_train, dtype=np.float32)
    clabel_eval = np.asarray(clabel_eval, dtype=np.float32)
    clabel_test = np.asarray(clabel_test, dtype=np.float32)

    sim_data_train = np.expand_dims(sim_data_train, axis=-1)
    sim_data_eval = np.expand_dims(sim_data_eval, axis=-1)
    sim_data_test = np.expand_dims(sim_data_test, axis=-1)

    sim_data = [sim_data_train, sim_data_eval, sim_data_test]
    extend = [extend_train, extend_eval, extend_test]
    label = [label_train, label_eval, label_test]
    clabel = [clabel_train, clabel_eval, clabel_test]

    for i in range(3):
        feature_list = [sim_data[i]]
        if add_temp:
            temp = extend[i][:, :, 0]
            temp = np.expand_dims(np.expand_dims(temp, axis=-1).repeat(32, axis=-1), axis=-1)
            feature_list.append(temp)
        if add_time_of_week:
            tow = extend[i][:, :, 1]
            tow = np.expand_dims(np.expand_dims(tow, axis=-1).repeat(32, axis=-1), axis=-1)
            feature_list.append(tow)

        processed_data = np.concatenate(feature_list, axis=-1)
        data = {}
        data["data"] = processed_data

        if i == 0:
            data["label"] = label[0]
            data["clabel"] = clabel[0]
            target = "train"
        elif i == 1:
            data["label"] = label[1]
            data["clabel"] = clabel[1]
            target = "eval"
        else:
            data["label"] = label[2]
            data["clabel"] = clabel[2]
            target = "test"
        with open(output_dir + "/Benchmark_{}.pkl".format(target), "wb") as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    INPUT_LEN = 128
    LABEL_LEN = 32
    NODE_NUM = 32
    DAMAGE_DEGREE = 3

    TARGET_CHANNEL = [0]  # target channel(s)
    STEPS_PER_WEEK = 28

    SOURCE_DATA_DIR = "Benchmark"
    DATASET_NAME = "Benchmark"
    TOW = True  # if add time_of_day feature
    TEMP = True  # if add temp feature
    OUTPUT_DIR = "src/data/" + DATASET_NAME
    DATA_FILE_PATH = "src/data/{}/Bridge".format(SOURCE_DATA_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Bridge data readings.")
    # parser.add_argument("--graph_file_path", type=str,
    #                     default=LABEL_FILE_PATH, help="Bridge data readings.")
    parser.add_argument("--input_len", type=int,
                        default=INPUT_LEN, help="Sequence Length.")
    parser.add_argument("--label_len", type=int,
                        default=LABEL_LEN, help="Sequence Length.")
    parser.add_argument("--node_num", type=int,
                        default=NODE_NUM, help="sensor num")
    parser.add_argument("--damage_degree", type=int,
                        default=DAMAGE_DEGREE, help="damage degree")
    parser.add_argument("--steps_per_week", type=int,
                        default=STEPS_PER_WEEK, help="Sequence Length.")
    parser.add_argument("--tow", type=bool, default=TOW,
                        help="Add feature time_of_week.")
    parser.add_argument("--temp", type=bool, default=TEMP,
                        help="Add feature temp.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--dataset_name", type=str,
                        default=DATASET_NAME, help="Dataset name")
    parser.add_argument("--split_case", type=bool,
                        default=True, help="is split case")
    parser.add_argument("--change_step", type=bool,
                        default=True, help="is change time step?")
    parser.add_argument("--add_gwn", type=bool,
                        default=False, help="is add noise?")
    parser.add_argument("--gwn", type=int,
                        default=20, help="SNR noise")

    args_metr = parser.parse_args()

    # print args
    print("-" * (20 + 45 + 5 + 6))
    for key, value in sorted(vars(args_metr).items()):
        print("|{0:>20} = {1:<51}|".format(key, str(value)))
    print("-" * (20 + 45 + 5 + 6))

    if not os.path.exists(args_metr.output_dir):
        os.makedirs(args_metr.output_dir)
    generate_noised_data(args_metr)
