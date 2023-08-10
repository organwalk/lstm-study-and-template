import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import build_model


# 数据预处理
def preprocess_data(df, date_Type):
    data = calculate_df_data_avg(df, date_Type)
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data, scaler


# 短期模型划分训练集输入、输出序列
def split_short_term_sequences(sequences):
    x, y = list(), list()
    scaler = None
    for i in range(len(sequences) - 1):
        # 读取当前和下一个数据
        df_current = pd.read_csv(sequences[i])
        df_next = pd.read_csv(sequences[i + 1])

        # 预处理数据
        data_current, scaler = preprocess_data(df_current, 'H')
        data_next, _ = preprocess_data(df_next, 'H')

        # 添加到输入和输出序列中
        x.append(data_current)
        y.append(data_next)

    return np.array(x), np.array(y), scaler


# 长期模型划分训练集输入、输出序列
def split_long_term_sequences(sequences, n_steps_in, n_steps_out):
    x, y = list(), list()
    scaler = None
    for i in range(len(sequences)):
        end_idx = i + n_steps_in
        out_end_idx = end_idx + n_steps_out
        if out_end_idx > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_idx], sequences[end_idx:out_end_idx]
        seq_x, scaler = preprocess_data(seq_x, 'D')
        seq_y, _ = preprocess_data(seq_y, 'D')
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y), scaler


# 获取所有CSV文件
def short_term_pre_process():
    files_data = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
    # 划分序列
    x, y, scaler = split_short_term_sequences(files_data)
    return x, y, scaler


# 预测时处理输入、输出序列的数据
def process_sequence_data(file, seq_type):
    df_data = pd.read_csv(file)
    if seq_type == 'input':
        return preprocess_data(df_data, 'H')
    elif seq_type == 'output':
        return calculate_df_data_avg(df_data, 'H')


def calculate_df_data_avg(df_data, date_type):
    df_data['Time'] = pd.to_datetime(df_data['Time'])
    df = df_data.set_index('Time')
    df_hourly = df.resample(date_type).mean().round(2).reset_index().drop('Time', axis=1)
    result = df_hourly.to_numpy().astype(int)
    return result


