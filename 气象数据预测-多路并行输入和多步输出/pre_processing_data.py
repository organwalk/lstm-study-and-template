import pandas as pd
from numpy import array


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


def get_data():
    # 从 CSV 文件中读取数据
    df = pd.read_csv('1_data_2023-06-29.csv')

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

    # 拆分数据为输入和输出序列
    n_steps_in = len(data) // 2
    n_steps_out = len(data) - n_steps_in

    x, y = split_sequences(data, n_steps_in, n_steps_out)

    n_features = x.shape[2]

    return x, y, n_features, n_steps_in, n_steps_out
