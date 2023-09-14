from meteo_analyze_predict_server.entity import res_entity
from config.application import get_mysql_obj


def model_info():
    mysql = get_mysql_obj()
    mysql.execute("SELECT * FROM model_info ORDER BY id DESC LIMIT 1")
    result = mysql.fetchall()
    return res_entity.model_info(*result[0][1:7]) if result else []
