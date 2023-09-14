from flask import Flask
from flask_cors import CORS
from meteo_analyze_predict_server.config import result
from config.application import register_to_nacos
import repository


app = Flask(__name__)
CORS(app)


@app.route('/anapredict/model/info', methods=['GET'])
def __api_model_info():
    info_data = repository.model_info()
    if info_data:
        return result.success('成功获取模型信息', info_data)
    else:
        return result.not_found('未能获取模型信息')


@app.errorhandler(404)
def __server_api_notfound(e):
    return result.not_found('该接口不存在，请修改后重试')


@app.errorhandler(500)
def __server_error(e):
    return result.error('内部服务处理错误')


if __name__ == '__main__':
    register_to_nacos()
    app.run(host='0.0.0.0', port=9594)
