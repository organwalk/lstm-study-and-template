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
    if model_type == 'ShortTermByLSTM':
        x, y, scaler = processing_data.short_term_pre_process()
        config = {
            'x': x,
            'y': y,
            'scaler': scaler,
            'steps_in': 24,
            'steps_out': 24,
            'features': 8,
            'evaluation': True,
            'date_type': 'H'
        }
        loading_short_term_model(config)
    elif model_type == 'LongTermWithinAWeekByLSTM':
        x, y = processing_data.long_term_with_in_a_week_pre_process(2, 1)
        config = {
            'x': x,
            'y': y,
            'steps_in': 2,
            'steps_out': 1,
            'features': 8,
            'evaluation': True,
            'date_type': 'D'
        }
        loading_long_term_with_in_a_week_model(config)


'''
    1.根据数据集训练模型
    2.处理前24小时的数据，作为输入序列
'''


def loading_short_term_model(config):
    model = build_model.training_short_term_model(
        config['x'], config['y'], config['steps_in'], config['steps_out'], config['features']
    )
    # model.save('meteo_model_short_term_oneday.h5')
    x_input, _ = processing_data.process_sequence_data('data/1_data_2023-06-28.csv', 'input', config['date_type'])
    x_input = x_input.reshape((1, config['steps_in'], config['features']))

    yhat = model.predict(x_input, verbose=1)

    yhat_round = np.round(yhat).astype(int)
    yhat_round_non_neg = np.clip(yhat_round, a_min=0, a_max=None).reshape((config['steps_out'], config['features']))

    yhat_inv = config['scaler'].inverse_transform(yhat_round_non_neg).reshape((config['steps_out'], config['features']))
    yhat_inv = np.round(yhat_inv).astype(int)

    yhat_inv_list = [list(row) for row in yhat_inv]

    print(yhat_inv_list)
    if config['evaluation']:
        evaluation_using_model(yhat_inv, config['date_type'])


def loading_long_term_with_in_a_week_model(config):
    model = build_model.training_long_term_with_in_a_week_model(
        config['x'], config['y'], config['steps_in'], config['steps_out'], config['features']
    )
    # model.save('meteo_model_short_term_oneday.h5')
    file_array = ['data/1_data_2023-06-27.csv', 'data/1_data_2023-06-28.csv']
    result_list = []  # 创建一个空列表来存储处理后的数据

    for file_path in file_array:
        x_input = processing_data.process_sequence_data(file_path, 'input', config['date_type'],
                                                        file_path.split('data_')[1][:10])
        result_list.append(x_input)  # 将处理后的数据添加到列表中

    x_input = np.array(result_list)  # 将列表转换为数组
    x_input = x_input.reshape((1, config['steps_in'], config['features']))

    yhat = model.predict(x_input, verbose=1)

    yhat_round = np.round(yhat).astype(int)
    yhat_round_non_neg = np.clip(yhat_round, a_min=0, a_max=None).reshape((config['steps_out'], config['features']))

    # yhat_inv = config['scaler'].inverse_transform(yhat_round_non_neg).reshape((config['steps_out'], config['features']))
    yhat_inv = np.round(yhat_round_non_neg).astype(int)

    yhat_inv_list = [list(row) for row in yhat_inv]
    print('++++++')
    print(yhat_inv_list)

    if config['evaluation']:
        evaluation_using_model(yhat_inv, config['date_type'])


def evaluation_using_model(yhat_inv, date_type):
    # 评估模型
    real_data = processing_data.process_sequence_data('data/1_data_2023-06-29.csv', 'output', date_type)
    real_data = np.reshape(real_data, (1, 24, 8))
    predict_data = np.reshape(yhat_inv, (1, 24, 8))
    evaluation_model.test_view_about_model(real_data, predict_data)
    evaluation_model.evaluate(real_data, predict_data)


start_predict('LongTermWithinAWeekByLSTM')
