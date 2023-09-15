"""
    封装响应中的实体
    by organwalk 2023-08-15
"""
from collections import OrderedDict


def model_info(version, cn_des, technology, support, update):
    """
    封装/anapredict/correlation接口的响应数据

    :param version: (str) 版本信息
    :param cn_des: (str) 中文描述
    :param technology: (str) 模型使用的技术栈
    :param support: (str) 支持的模型类别
    :param update: (str) 最后一次更新日期
    :return:
        dict: model_info的有序字典

    by organwalk 2023-08-15
    """
    return OrderedDict([
        ('version', version),
        ('cn_des', cn_des),
        ('technology', technology),
        ('support', support),
        ('update', update)
    ])
