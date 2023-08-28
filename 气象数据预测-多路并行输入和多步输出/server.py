from flask import Flask, jsonify, request
from flask_cors import CORS
from meteo_model import start_predict
import result
import numpy as np
import pandas as pd
app = Flask(__name__)
CORS(app)


@app.route('/qx/model/predict', methods=['POST'])
def http_model_predict():
    required_fields = ['station', 'start_date', 'end_date', 'model_type']
    validation_error = result.validate_fields(request.get_json(), required_fields)
    if validation_error:
        return jsonify(validation_error)
    else:
        predict_result = start_predict(request.get_json())
    if predict_result is not None:
        if predict_result['status']:
            return jsonify(result.success(predict_result['msg'], predict_result['data']))
        else:
            return jsonify(result.error(404, predict_result['msg']))
    else:
        return jsonify(result.error(500, "内部服务错误"))


@app.route('/qx/model/info', methods=['GET'])
def http_model_info():
    data = {
        'version': '#22803202902C',
        'cn_des': '信创技术下气象数据预测模型',
        'technology': '基于TensorFlow、StatsModels和Prophet构建预测模型',
        'support': 'LSTM、ARIMA、Prophet',
        'update': '2023-08-25'
    }
    return jsonify(result.success('获取成功', data))


@app.route('/qx/model/list')
def http_model_list():
    data = {
        'modelList': ['LSTM', 'ARIMA', 'Prophet', 'Mixed Models']
    }
    return jsonify(data)


@app.route('/qx/correlation', methods=['POST'])
def http_model_correlation():
    required_fields = ['station', 'start_date', 'end_date', 'correlation']
    missing_fields = []
    for field in required_fields:
        if field not in request.args:
            missing_fields.append(field)
    if missing_fields:
        if len(missing_fields) > 1:
            message = f"以下字段不能为空: {', '.join(missing_fields)}"
        else:
            message = f"字段 '{missing_fields[0]}' 不能为空"
        return jsonify({'error': message}), 400
    else:
        data = pd.read_csv('data//1_data_2023-06-27.csv')

        columns = ['Temperature', 'Humidity', 'Speed', 'Direction', 'Rain', 'Sunlight', 'PM2.5', 'PM10']
        data_subset = data[columns]

        data_array = data_subset.to_numpy()

        correlation_matrix = np.corrcoef(data_array, rowvar=False)
        correlation_matrix[np.isnan(correlation_matrix)] = 0
        result_array = correlation_matrix.tolist()
        print(result_array)

        # 执行你的相关操作
        return jsonify({'data': '200'})


@app.errorhandler(405)
def handle_method_not_allowed(e):
    url_rule = next(
        (rule for rule in app.url_map.iter_rules() if 'qx/predict' in rule.rule),
        None
    )
    allowed_methods = [method for method in url_rule.methods if method != 'OPTIONS'] if url_rule else []
    return jsonify(result.error(405, f"该接口仅支持{', '.join(allowed_methods)}方法"))


@app.errorhandler(500)
def server_error(e):
    return jsonify(result.error(500, '内部服务错误'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9594)
