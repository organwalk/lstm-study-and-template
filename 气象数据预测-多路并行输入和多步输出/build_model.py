from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from evaluation_model import get_chart_about_training_loss
import numpy as np


'''
    存在过拟合现象，考虑以下解决方案：
    1.加大数据量
    2.对训练数据采用随即变换、旋转、缩放等操作进行数据增强，增加数据多样性（不优先考虑）
    3.减小模型复杂度（调参），增加正则化技术，限制模型参数大小：
        L1和L2正则化。
            L1正则化有助于特征选择和模型的解释性，因为它可以使得某些特征对模型的预测贡献为0，从而识别出对预测不重要的特征
            L2正则化有助于控制模型的复杂性，使得权重分布更加平滑，有助于减少过拟合的风险
        Dropout。随机丢弃一部分神经元，降低过拟合风险
      √ EarlyStopping。验证集损失函数停止改善时停止训练
'''


def training_short_term_model(x, y, n_steps_in, n_steps_out, n_features):
    model = Sequential()

    model.add(LSTM(500, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(500, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))

    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    history = model.fit(x, y, epochs=300, verbose=1, validation_split=0.2, callbacks=[early_stopping])

    get_chart_about_training_loss(history)

    return model


def training_long_term_with_in_a_week_model(x, y, n_steps_in, n_steps_out, n_features):
    model = Sequential()

    model.add(LSTM(500, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(500, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))

    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    history = model.fit(x, y, epochs=10, verbose=1)

    # get_chart_about_training_loss(history)

    # 逐步预测
    predictions = []
    current_input = x[0]  # 使用第一个输入数据开始预测

    for _ in range(n_steps_out):
        # 预测一个时间步
        predicted_step = model.predict(np.array([current_input]))[0]
        print(predicted_step)
        predictions.append(predicted_step)

        # 更新当前输入序列，将刚刚预测的结果加入到输入序列中
        current_input = np.concatenate((current_input[1:], predicted_step[-1:]), axis=0)

    return model
