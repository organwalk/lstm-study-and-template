"""
    定义调用方请求接口校验的方法，返回错误消息，若校验通过则返回None
    by organwalk 2023-08-15
"""
import re
from meteo_analyze_predict_server.repository import repository


def json_user_req_validate(api, user_req_json, server_req_fields):
    """
        校验调用方请求中的 JSON 数据

        Args:
            api (str): API 接口路径
            user_req_json (dict): 调用方请求中的 JSON 数据
            server_req_fields (list): 服务器端要求的 JSON 数据

        Returns:
            str or None: 错误消息，如果校验通过则返回 None

        Execution:
            1.检验调用方请求是否存在JSON格式数据
            2.检验JSON格式数据是否存在空缺值
            3.检验对应接口的JSON格式数据其值类型与格式是否正确

        by organwalk 2023-08-15
    """
    if not user_req_json:
        return '调用方未能正确传递JSON格式数据'

    missing_check = __json_missing_validate(user_req_json, server_req_fields)
    if missing_check:
        return '调用方传递JSON格式数据存在空缺值，提示消息如下：' + missing_check

    error_msg = __json_value_validate(user_req_json, api)
    if error_msg is not None:
        return '调用方传递JSON格式数据存在错误，提示消息如下：' + error_msg

    return None


def __json_missing_validate(user_req_json, server_req_fields):
    """
        校验调用方请求中的 JSON 数据是否含有空缺值

        Args:
        user_req_json (dict): 调用方请求中的 JSON 数据
        server_req_fields (list): 服务器端要求的 JSON 数据

        Returns:
        str or None: 错误消息，如果校验通过则返回 None

        Execution:
        1.将服务端要求的字段与调用方JSON字段进行比较
        2.将空缺字段取出并使用顿号分割形成错误消息语句

        by organwalk 2023-08-15
    """
    missing_fields = [field for field in server_req_fields if not user_req_json.get(field)]
    if missing_fields:
        missing_fields_str = "、".join(missing_fields)
        return f"字段{missing_fields_str}不能为空"
    return None


def __json_value_validate(user_req_json, api):
    """
    校验调用方JSON数据的类型与格式是否正确

    :param user_req_json(str): 调用方传递的JSON数据
    :param api(str): API接口路径
    :return:
        str or None: 错误消息，如果校验通过则返回None

    by organwalk 2023-08-15
    """
    if api == '/anapredict/correlation':
        return __api_correlation_validate(user_req_json)
    else:
        return None


def __api_correlation_validate(user_req_json):
    """
    校验/anapredict/correlation接口的JSON数据

    :param user_req_json(str): 调用方传递的JSON数据
    :return:
        str or None: 错误消息，如果校验通过则返回None

    by organwalk 2023-08-15
    """
    error_msg_list = [__station_error(user_req_json['station']),
                      __start_end_date_error(user_req_json['station'], user_req_json['start_date']),
                      __start_end_date_error(user_req_json['station'], user_req_json['end_date']),
                      __which_or_correlation_error(user_req_json['correlation'])]
    msg_list = [msg['msg'] for msg in error_msg_list if isinstance(msg, dict)]
    return '；'.join(set(msg_list)) if msg_list else None


__YYYY_MM_DD_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
__METEO_ELEMENTS_PATTERN = r'^[1-8,]+$'


def __station_error(station):
    """
    校验气象站编号字段的正确性
    :param station(str): 气象站编号
    :return:
        str or dict: station的值，如果校验不通过则返回错误消息

    by organwalk 2023-08-15
    """
    return station if isinstance(station, str) else __error_msg("station字段需要字符串类型的气象站编号")


def __start_end_date_error(station, date):
    """
    校验起始日期和截止日期的有效性
    :param station(str): 气象站编号
    :param date(str): 日期
    :return:
        str or dict: date的值，如果校验不通过则返回错误消息

    by organwalk 2023-08-15
    """
    if isinstance(date, str) and re.match(__YYYY_MM_DD_PATTERN, date):
        if repository.station_valid_date(station, date) > 0:
            return date
        else:
            return __error_msg("值为" + date + "的date相关字段，其日期下不存在有效数据，请重新指定")
    else:
        return __error_msg("date相关字段需要YYYY-MM-DD格式字符串")


def __which_or_correlation_error(elements):
    """
    校验气象要素的正确性
    :param elements(str): 气象要素
    :return:
        str or dict: elements的值，如果校验不通过则返回错误消息

    by organwalk 2023-08-15
    """
    if isinstance(elements, str) and re.match(__METEO_ELEMENTS_PATTERN, elements):
        numbers = [int(num) for num in elements.split(',') if num.isdigit()]
        if all(1 <= num <= 8 for num in numbers):
            return elements
    return __error_msg("气象要素相关字段需要以英文逗号分割的数字字符串格式数据，例如：1,2,3,范围在1-8")


def __error_msg(msg):
    """
    返回字典形式的错误消息

    :param msg(str): 错误消息
    :return:
        dict: 错误消息的字典形式

    by organwalk 2023-08-15
    """
    return {'msg': msg}
