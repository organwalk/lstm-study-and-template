from meteo_analyze_predict_server.entity import res_entity
from meteo_analyze_predict_server.config.application import mysql_obj
import meteo_analyze_predict_server.repository.mysql_statements as mysql_statements


def model_info():
    mysql = mysql_obj()
    mysql.execute(mysql_statements.NEW_MODEL_INFO)
    result = mysql.fetchall()[0][1:7]
    return res_entity.model_info(*result) if result else []


def station_valid_date(station, date):
    mysql = mysql_obj()
    mysql.execute(mysql_statements.VALID_DATE, (station, date))
    return mysql.fetchall()[0][0]

