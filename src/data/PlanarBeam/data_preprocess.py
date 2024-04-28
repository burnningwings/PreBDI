import warnings
warnings.filterwarnings('ignore')
 
import seaborn as sns
import matplotlib.pyplot as plt
 
import pandas as pd
import numpy as np
import math
from random import randrange
 
pd.options.display.max_columns  = 100

from sklearn.preprocessing import StandardScaler
def nor_malize(x):
    scaler  = StandardScaler().fit(x)
    x_scaler = scaler.transform(x)   
    return x_scaler

from sklearn.preprocessing import MinMaxScaler
def max_min(x):
    scaler  = MinMaxScaler().fit(x)
    x_scaler = scaler.transform(x)   
    return x_scaler

def add_noise(data):
    import random
    mu = 0
    sigma = 0.1
    data = data + np.random.normal(mu,sigma,data.shape)
    return data

label_level = 10
non_damage_len = 4464

# 测试集时间长度 - 1
train_time_len = math.ceil(non_damage_len * 0.8)
eval_time_len = math.ceil(non_damage_len * 0.9) - train_time_len
test_time_len = non_damage_len - train_time_len - eval_time_len

# 在这里选择传感器
cols = ['s' + str(i) for i in range(1,21)]
unremove_cols = ['s' + str(j) for j in range(1,21)]


removed_col = []
# # 删除10%
# for i in range(2):
#     removed_col.append(unremove_cols.pop(randrange(len(unremove_cols)))) 
# 删除20%
for i in range(6):
    removed_col.append(unremove_cols.pop(randrange(len(unremove_cols)))) 

for i in ["train", "eval", "test"]:
    time_len = train_time_len
    if i == "train":
        time_len = train_time_len
    elif i == "eval":
        time_len = eval_time_len
    else:
        time_len = test_time_len
    data_path = "/mnt/data/home/liujun/afinal-bishe/1_multi/data_csv/multi_3degree_20sensor_singlemouth/multi_3degree_20sensor_timesplited_"
    data = pd.read_csv(data_path + i + ".csv")

    # for j in removed_col:
    #     data[j] = 0

    # 要求对缺省传感器置0
    unremove_no_damage = data[:time_len][unremove_cols][-127:]
    
    no_damage = data[:time_len][cols][-127:]
    label_cols = ['loc1','loc2','loc3','quant1','quant2','quant3']
    tempo_cols = ['tempo','time_in_day','day_in_mouth']
    no_damage_tempo = data[:time_len][tempo_cols][-127:]

    from tqdm import tqdm
    import random
    random.seed(2022)
    if i == "train":
        list_j = sorted(random.sample(range(128,128+time_len),100))
    elif i == "eval":
        list_j = sorted(random.sample(range(128,128+time_len),15))
    else:
        list_j = sorted(random.sample(range(128,128+time_len),15))
    
    lsit_empty = []
    label = []
    sim_data = []
    # 要求对缺省传感器置0
    unremove_sim_data = []

    tempo = []

    for n in tqdm(range(1,data.shape[0]//time_len)):
        
        df_label = data[time_len*n:time_len*(n+1)][label_cols]
        df_tempo = data[time_len*n:time_len*(n+1)][tempo_cols]
        list_label = [0 for j in range(250)]

        l1 = df_label['loc1'][time_len*n]
        q1 = df_label['quant1'][time_len*n] / 100
        list_label[l1 - 1] = q1

        l2 = df_label['loc2'][time_len*n]
        q2 = df_label['quant2'][time_len*n] / 100
        list_label[l2 - 1] = q2

        l3 = df_label['loc3'][time_len*n]
        q3 = df_label['quant3'][time_len*n] / 100
        if l3 != 0:
            list_label[l3 - 1] = q3
    


        df_damage = data[time_len*n:time_len*(n+1)][cols]
        df_damage = pd.concat([no_damage,df_damage]).reset_index(drop=True)
        # 要求对缺省传感器置0
        unremove_df_damage = data[time_len*n:time_len*(n+1)][unremove_cols]
        unremove_df_damage = pd.concat([unremove_no_damage, unremove_df_damage]).reset_index(drop=True)
        
        df_tempo = pd.concat([no_damage_tempo,df_tempo]).reset_index(drop=True)
        for j in list_j:

            sn_data = df_damage.values[j-127:j+1]
            
            # 要求对缺省传感器置0
            unremove_sn_data = unremove_df_damage.values[j-127:j+1]
            tempo_data = df_tempo.values[j-127:j+1]
            
            sim_data.append(sn_data)
            
            # 要求对缺省传感器置0
            unremove_sim_data.append(unremove_sn_data)
            label.append(list_label)
            tempo.append(tempo_data)
        

    label = np.asarray(label, dtype= np.float32)

    sim_data = np.asarray(sim_data, dtype= np.float32)
    
    # 要求对缺省传感器置0
    unremove_sim_data = np.asarray(unremove_sim_data, dtype = np.float32)

    tempo = np.asarray(tempo, dtype= np.float32)
    sim_data = sim_data.reshape(sim_data.shape[0]*sim_data.shape[1],sim_data.shape[2])
    unremove_sim_data = unremove_sim_data.reshape(unremove_sim_data.shape[0]*unremove_sim_data.shape[1], unremove_sim_data.shape[2])
    tempo = tempo.reshape(tempo.shape[0]*tempo.shape[1],tempo.shape[2])


    # remove = []
    for j in removed_col:
        sim_data[:,int(j[1:])-1] = 0
    
    sim_data = nor_malize(sim_data)
    unremove_sim_data = nor_malize(unremove_sim_data)
    # sim_data = add_noise(sim_data)

    time_step = 128
    n_feature = 18
    sim_data = sim_data.reshape(-1, time_step, len(cols))
    unremove_sim_data = unremove_sim_data.reshape(-1, time_step, len(unremove_cols))
    tempo = tempo.reshape(-1,time_step,3)

    save_path = "/mnt/data/home/liujun/afinal-bishe/1_multi/data/multi_3degree_20sensor_singlemouth_remove30%/"
    data_x_path = save_path + "feature_" + i + ".npy"
    data_y_path = save_path + "label_" + i + ".npy"
    data_z_path = save_path + "tempo_" + i + ".npy"
    removed_data_x_path = save_path + "unremove_feature_" + i + ".npy"
    np.save(data_x_path,sim_data)
    np.save(removed_data_x_path,unremove_sim_data)
    np.save(data_y_path ,label)
    np.save(data_z_path ,tempo)

    x_all = np.load(data_x_path)
    x_all = x_all.reshape(x_all.shape[0],x_all.shape[1],len(cols),1)
    y_all = np.load(data_y_path)
    y_all = y_all * 10 * label_level
    z_all = np.load(data_z_path)
    z_all = np.tile(z_all,(1,1,len(cols)))
    z_all = z_all.reshape(z_all.shape[0],z_all.shape[1],len(cols),3)
    x_all = np.concatenate((x_all,z_all),axis=-1)

    np.save(save_path + "feature_" + i + "_STID.npy",x_all)
    np.save(save_path + "label_" + i + "_STID.npy",y_all)

