from flask import Flask, jsonify, request
from meteo_model import start_predict
import result
app = Flask(__name__)


@app.route('/qx/predict', methods=['POST'])
def endpoint_a():
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



@app.route('/b')
def endpoint_b():
    data = {'data': 'This is endpoint B'}
    return jsonify(data)


@app.route('/c')
def endpoint_c():
    data = {'data': 'This is endpoint C'}
    return jsonify(data)


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
    app.run(port=9594)
