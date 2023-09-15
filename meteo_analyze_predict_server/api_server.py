"""
    Flask应用接口服务，同时将服务注册到nacos中
    by organwalk 2023-08-15
"""
from flask import Flask, request
from flask_cors import CORS
from meteo_analyze_predict_server.config import result
from config.application import register_to_nacos
from meteo_analyze_predict_server.repository import repository
import entity.req_entity as server_req
from utils import req_utils


app = Flask(__name__)
CORS(app)


@app.route('/anapredict/model/info', methods=['GET'])
def __api_model_info():
    """
    从MySQL数据库中获取并返回模型信息

    :return:
        json: 根据获取状态返回相应的消息以及数据

    by organwalk 2023-08-15
    """
    info_data = repository.model_info()
    return result.success('成功获取模型信息', info_data) if info_data else result.not_found('未能获取模型信息')


@app.route('/anapredict/correlation', methods=['POST'])
def __api_data_correlation():
    """
    对给定的要求进行气象数据分析

    :return:
        json: 根据获取状态返回相应的消息以及数据

    by organwalk 2023-08-15
    """
    validate = req_utils.json_user_req_validate('/anapredict/correlation',
                                                request.get_json(),
                                                server_req.CORRELATION)
    if validate is None:
        return result.success("成功分析", "数据")
    else:
        return result.fail_entity(validate)


@app.errorhandler(404)
def __server_api_notfound(e):
    return result.not_found('该接口不存在，请修改后重试')


@app.errorhandler(500)
def __server_error(e):
    return result.error('内部服务处理错误')


if __name__ == '__main__':
    register_to_nacos()
    app.run(host='0.0.0.0', port=9594)
