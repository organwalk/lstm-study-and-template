import pandas as pd
from numpy import array
from sklearn.preprocessing import MinMaxScaler


def split_sequences(sequences, n_steps_in, n_steps_out):
    x, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        x.append(seq_x)
        y.append(seq_y)

    return array(x), array(y)


def split_other_day_sequences(input_seq, output_seq):
    x, y = list(), list()
    x.append(input_seq)
    y.append(output_seq)
    return array(x), array(y)


def get_data_same_day():
    # 从 CSV 文件中读取数据
    df = pd.read_csv('1_data_2023-06-25.csv')

    # 将时间戳列转换为 DatetimeIndex 对象
    df['Time'] = pd.to_datetime(df['Time'])

    # 将 DatetimeIndex 对象设置为索引
    df = df.set_index('Time')

    # 将数据按小时汇总并计算平均值
    df_hourly = df.resample('H').mean()
    df_hourly = df_hourly.round(2)
    df_hourly = df_hourly.reset_index()
    df_hourly = df_hourly.drop('Time', axis=1)
    data = df_hourly.to_numpy()
    data = data.astype(int)
    print(data)
    # 归一化处理
    scaler_out = MinMaxScaler()
    scaler_out.fit(data)
    data = scaler_out.transform(data)

    # 拆分数据为输入和输出序列
    n_steps_in = len(data) // 2
    n_steps_out = len(data) - n_steps_in

    x, y = split_sequences(data, n_steps_in, n_steps_out)

    n_features = x.shape[2]

    return x, y, n_features, n_steps_in, n_steps_out, scaler_out


def get_data_other_day():
    # 从 CSV 文件中读取数据
    df_in = pd.read_csv('1_data_2023-06-27.csv')
    df_out = pd.read_csv('1_data_2023-06-28.csv')

    # 将时间戳列转换为 DatetimeIndex 对象
    df_in['Time'] = pd.to_datetime(df_in['Time'])
    df_out['Time'] = pd.to_datetime(df_out['Time'])

    # 将 DatetimeIndex 对象设置为索引
    df_in = df_in.set_index('Time')
    df_out = df_out.set_index('Time')

    # 将数据按小时汇总并计算平均值
    df_in_hourly = df_in.resample('H').mean()
    df_out_hourly = df_out.resample('H').mean()
    df_in_hourly = df_in_hourly.round(2)
    df_out_hourly = df_out_hourly.round(2)
    df_in_hourly = df_in_hourly.reset_index()
    df_out_hourly = df_out_hourly.reset_index()
    df_in_hourly = df_in_hourly.drop('Time', axis=1)
    df_out_hourly = df_out_hourly.drop('Time', axis=1)
    data_in = df_in_hourly.to_numpy()
    data_out = df_out_hourly.to_numpy()
    data_in = data_in.astype(int)
    data_out = data_out.astype(int)

    # 归一化处理输入数据
    scaler_in = MinMaxScaler()
    scaler_in.fit(data_in)
    data_in = scaler_in.transform(data_in)

    # 归一化处理输出数据
    scaler_out = MinMaxScaler()
    scaler_out.fit(data_out)
    data_out = scaler_out.transform(data_out)

    # 拆分数据为输入和输出序列
    n_steps_in = len(data_in)
    n_steps_out = len(data_out)

    x, y = split_other_day_sequences(data_in, data_out)

    n_features = x.shape[2]

    return x, y, n_features, n_steps_in, n_steps_out, scaler_out
