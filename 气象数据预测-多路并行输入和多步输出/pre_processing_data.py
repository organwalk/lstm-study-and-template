import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import build_model


# 数据预处理
def preprocess_data(df):
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time')
    df_hourly = df.resample('H').mean()
    df_hourly = df_hourly.round(2)
    df_hourly = df_hourly.reset_index()
    df_hourly = df_hourly.drop('Time', axis=1)
    data = df_hourly.to_numpy()
    data = data.astype(int)
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data, scaler


# 短期模型划分训练集
def split_short_term_sequences(sequences):
    x, y = list(), list()
    scaler = None
    for i in range(len(sequences) - 1):
        # 读取当前和下一个数据
        df_current = pd.read_csv(sequences[i])
        df_next = pd.read_csv(sequences[i + 1])

        # 预处理数据
        data_current, scaler = preprocess_data(df_current)
        data_next, _ = preprocess_data(df_next)

        # 添加到输入和输出序列中
        x.append(data_current)
        y.append(data_next)

    return np.array(x), np.array(y), scaler


# 获取所有CSV文件的文件名
def oneday_pre_process():
    path = "data"
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    # 划分序列
    x, y, scaler = split_short_term_sequences(files)
    return x, y, scaler


def gave_data_need_to_process(file, type):
    df_data = pd.read_csv(file)
    if type == 'input':
        return preprocess_data(df_data)
    elif type == 'output':
        df_data['Time'] = pd.to_datetime(df_data['Time'])
        df = df_data.set_index('Time')
        df_hourly = df.resample('H').mean()
        df_hourly = df_hourly.round(2)
        df_hourly = df_hourly.reset_index()
        df_hourly = df_hourly.drop('Time', axis=1)
        data = df_hourly.to_numpy()
        data = data.astype(int)
        return data

