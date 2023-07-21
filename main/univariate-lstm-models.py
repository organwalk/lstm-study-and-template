from numpy import array

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)


# 样本数据
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# 选择时间步长
n_steps = 3
x, y = split_sequence(raw_seq, n_steps)
for i in range(len(x)):
    print(x[i], y[i])
