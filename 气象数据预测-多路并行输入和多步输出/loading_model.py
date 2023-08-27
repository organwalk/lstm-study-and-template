import numpy as np
import build_model
import processing_data
import evaluation_model
from tensorflow.keras.models import load_model


def short_term_model(config):
    missing_dates_length = len(config['missing_dates'])
    existing_files_length = len(config['existing_file'])
    if missing_dates_length == 0 and existing_files_length != 0:
        status = True
        msg = '成功获取预测结果'
    elif existing_files_length == 0:
        status = False
        msg = f"由于以下日期：{config['missing_dates']}其本身及其邻近范围内未能获取到有效数据文件，模型无法对此采用【向前填充】处理，故无法进行预测"
        return {
            'status': status,
            'msg': msg,
        }
    else:
        status = True
        msg = f"以下日期未能获取到有效数据文件：{config['missing_dates']}，模型将对此采用【向前填充】处理，预测结果可能受此影响"
    # model = build_model.short_term(
    #     config['x'], config['y'], config['steps_in'], config['steps_out'], config['features']
    # )
    model = load_model('short_term.h5')
    result_list = []
    scaler = ''
    for file_path in config['existing_file']:
        x_input, scaler = processing_data.sequence(file_path, 'input', config['date_type'])
        result_list.append(x_input)
    x_input = np.array(result_list)
    print(x_input)
    config['scaler'] = scaler
    prediction_list = predict(x_input, config, model)

    return {
        'status': status,
        'msg': msg,
        'data': prediction_list
    }


def long_term_with_in_a_week_model(config):
    missing_dates_length = len(config['missing_dates'])
    existing_files_length = len(config['existing_files'])
    if missing_dates_length == 0 and existing_files_length != 0:
        status = True
        msg = '成功获取预测结果'
    elif existing_files_length == 0:
        status = False
        msg = f"由于以下日期：{config['missing_dates']}其本身及其邻近范围内未能获取到有效数据文件，模型无法对此采用【向前填充】处理，故无法进行预测"
        return {
            'status': status,
            'msg': msg,
        }
    else:
        status = True
        msg = f"以下日期未能获取到有效数据文件：{config['missing_dates']}，模型将对此采用【向前填充】处理，预测结果可能受此影响"
    model = build_model.long_term_with_in_a_week(
        config['x'], config['y'], config['steps_in'], config['steps_out'], config['features']
    )

    # file_array = config['existing_files']
    file_array = ['data\\1_data_2023-06-27.csv', 'data\\1_data_2023-06-28.csv']
    result_list = []
    for file_path in file_array:
        x_input = processing_data.sequence(file_path, 'input', config['date_type'], file_path.split('data_')[1][:10])
        result_list.append(x_input)
    x_input = np.array(result_list)
    prediction_list = predict(x_input, config, model)

    return {
        'status': status,
        'msg': msg,
        'data': prediction_list
    }


def predict(x_input, config, model):
    print(x_input)
    x_input = x_input.reshape((1, config['steps_in'], config['features']))

    yhat = model.predict(x_input, verbose=1)

    yhat_round = np.round(yhat).astype(int)
    yhat_round_non_neg = np.clip(yhat_round, a_min=0, a_max=None).reshape((config['steps_out'], config['features']))

    if 'scaler' in config:
        yhat_inv = config['scaler'].inverse_transform(yhat_round_non_neg).reshape(
            (config['steps_out'], config['features']))
        yhat_inv = np.round(yhat_inv).astype(int)
    else:
        yhat_inv = np.round(yhat_round_non_neg).astype(int)

    if config['evaluation']:
        evaluation_model.start(yhat_inv, config['date_type'])

    # yhat_inv_list = [list(row) for row in yhat_inv]

    return yhat_inv.tolist()
