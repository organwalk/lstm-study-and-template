import pymysql
from collections import OrderedDict
from nacos import NacosClient
import threading
import time


__mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'meteo_data'
}


def get_mysql_obj():
    cnx = pymysql.connect(**__mysql_config)
    return cnx.cursor()


__nacos_config = OrderedDict([
    ('service_name', 'meteo-anapredict-resource'),
    ('ip', 'localhost'),
    ('port', 9594),
    ('weight', 1.0)
])


def send_heartbeat_periodically(client):
    first_run = True
    while True:
        first_run = False if first_run else time.sleep(10)
        client.send_heartbeat(**__nacos_config)


def register_to_nacos():
    client = NacosClient('localhost:8848')
    client.add_naming_instance(**__nacos_config)
    heartbeat_thread = threading.Thread(target=send_heartbeat_periodically, args=(client,), daemon=True)
    heartbeat_thread.start()



