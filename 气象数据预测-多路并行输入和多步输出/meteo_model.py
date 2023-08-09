import numpy as np

from keras.models import load_model
import evaluation_model
import pre_processing_data
import build_model

'''
    1.数据预处理
    2.设置输入和输出序列的时间步长为24，基于24小时的时间预测
    3.设置特征值为8，即有8个气象要素
'''
x, y, scaler = pre_processing_data.oneday_pre_process()
n_steps_in = 24
n_steps_out = 24
n_features = 8

'''
    1.根据数据集训练模型
    2.处理前24小时的数据，作为输入序列
'''
model = build_model.training_short_term_model_about_oneday(x, y, n_steps_in, n_steps_out, n_features)
# model.save('meteo_model_short_term_oneday.h5')
x_input, _ = pre_processing_data.gave_data_need_to_process('data/1_data_2023-06-28.csv', 'input')
x_input = x_input.reshape((1, n_steps_in, n_features))


yhat = model.predict(x_input, verbose=1)

yhat_round = np.round(yhat).astype(int)
yhat_round_non_negative = np.clip(yhat_round, a_min=0, a_max=None)
yhat_round_non_negative = yhat_round_non_negative.reshape((n_steps_out, n_features))

yhat_inv = scaler.inverse_transform(yhat_round_non_negative)
yhat_inv = yhat_inv.reshape((n_steps_out, n_features))
yhat_inv = np.round(yhat_inv).astype(int)

yhat_inv_list = [list(row) for row in yhat_inv]

print(yhat_inv_list)
# 评估模型
real_data = pre_processing_data.gave_data_need_to_process('data/1_data_2023-06-29.csv', 'output')
real_data = np.reshape(real_data, (1, 24, 8))
predict_data = np.reshape(yhat_inv, (1, 24, 8))
evaluation_model.test_view_about_model(real_data, predict_data)
evaluation_model.evaluate(real_data, predict_data)


# model = load_model('model_file_v0.1.h5')
#
