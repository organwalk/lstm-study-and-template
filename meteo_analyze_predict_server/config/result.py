"""
    定义响应处理
    成功：200
    未找到：404
    错误：500
    未通过字段检查：422
    by organwalk 2023-08-15
"""
from collections import OrderedDict
from flask import Response
import json


def success(msg, data):
    return Response(
        json.dumps(
            OrderedDict([
                ('code', 200),
                ('msg', msg),
                ('data', data)
            ])
        ),
        mimetype='application/json'
    )


def not_found(msg):
    return Response(
        json.dumps(
            OrderedDict([
                ('code', 404),
                ('msg', msg)
            ])
        ),
        mimetype='application/json'
    )


def error(msg):
    return Response(
        json.dumps(
            OrderedDict([
                ('code', 500),
                ('msg', msg)
            ])
        ),
        mimetype='application/json'
    )


def fail_entity(msg):
    return Response(
        json.dumps(
            OrderedDict([
                ('code', 422),
                ('msg', msg)
            ])
        ),
        mimetype='application/json'
    )
