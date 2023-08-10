import numpy as np

from keras.models import load_model
import evaluation_model
import processing_data
import build_model

'''
    1.数据预处理
    2.设置输入和输出序列的时间步长为24，基于24小时的时间预测
    3.设置特征值为8，即有8个气象要素
'''


def start_predict(model_type):
    if model_type == 'short':
        config = {
            'x': processing_data.short_term_pre_process()[0],
            'y': processing_data.short_term_pre_process()[1],
            'scaler': processing_data.short_term_pre_process()[2],
            'steps_in': 24,
            'steps_out': 24,
            'features': 8,
            'evaluation': True
        }
        loading_model(config)


'''
    1.根据数据集训练模型
    2.处理前24小时的数据，作为输入序列
'''


def loading_model(config):
    model = build_model.training_short_term_model(
        config['x'], config['y'], config['steps_in'], config['steps_out'], config['features']
    )
    # model.save('meteo_model_short_term_oneday.h5')
    x_input, _ = processing_data.process_sequence_data('data/1_data_2023-06-28.csv', 'input')
    x_input = x_input.reshape((1, config['steps_in'], config['features']))

    yhat = model.predict(x_input, verbose=1)

    yhat_round = np.round(yhat).astype(int)
    yhat_round_non_neg = np.clip(yhat_round, a_min=0, a_max=None).reshape((config['steps_out'], config['features']))

    yhat_inv = config['scaler'].inverse_transform(yhat_round_non_neg).reshape((config['steps_out'], config['features']))
    yhat_inv = np.round(yhat_inv).astype(int)

    yhat_inv_list = [list(row) for row in yhat_inv]

    print(yhat_inv_list)
    if config['evaluation']:
        evaluation_using_model(yhat_inv)


def evaluation_using_model(yhat_inv):
    # 评估模型
    real_data = processing_data.process_sequence_data('data/1_data_2023-06-29.csv', 'output')
    real_data = np.reshape(real_data, (1, 24, 8))
    predict_data = np.reshape(yhat_inv, (1, 24, 8))
    evaluation_model.test_view_about_model(real_data, predict_data)
    evaluation_model.evaluate(real_data, predict_data)


start_predict('short')
