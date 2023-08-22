import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 数据预处理
def preprocess_data(df, date_type, file_date=None):
    if file_date is not None:
        data = calculate_df_data_avg(df, date_type, file_date)
        return data[0]
    else:
        data = calculate_df_data_avg(df, date_type)
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
    data = []
    for file in sequences:
        df = pd.read_csv(file)
        processed_data = preprocess_data(df, 'D', file.split('data_')[1][:10])
        data.append(processed_data)
    x, y = [], []
    for i in range(len(data)):
        # 划分输入序列
        end_ix = i + n_steps_in
        # 划分输出序列
        out_end_ix = end_ix + n_steps_out
        # 确保划分的结束索引不超过数据范围
        if out_end_ix <= len(data):
            # 提取输入和输出序列
            seq_x = data[i:end_ix]
            seq_y = data[end_ix:out_end_ix]

            x.append(seq_x)
            y.append(seq_y)
    print(np.array(x))
    print(np.array(y))
    return np.array(x), np.array(y)


# 短期模型数据预处理
def short_term_pre_process():
    files_data = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
    x, y, scaler = split_short_term_sequences(files_data)
    return x, y, scaler


# 七天内长期模型数据预处理
def long_term_with_in_a_week_pre_process(n_steps_in, n_steps_out):
    files_data = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
    x, y = split_long_term_sequences(files_data, n_steps_in, n_steps_out)
    return x, y


# 预测时处理输入、输出序列的数据
def process_sequence_data(file, seq_type, date_type, file_date=None):
    df_data = pd.read_csv(file)
    if seq_type == 'input':
        if file_date is not None:
            return preprocess_data(df_data, date_type, file_date)
        else:
            return preprocess_data(df_data, date_type)
    elif seq_type == 'output':
        return calculate_df_data_avg(df_data, date_type)


def calculate_df_data_avg(df_data, date_type, custom_time=None):
    if custom_time is not None:
        df_data.loc[:, 'Time'] = pd.to_datetime(custom_time)
    else:
        df_data['Time'] = pd.to_datetime(df_data['Time'])
    df = df_data.set_index('Time')
    df_hourly = df.resample(date_type).mean().round(2).reset_index().drop('Time', axis=1)
    result = df_hourly.to_numpy().astype(int)
    return result
