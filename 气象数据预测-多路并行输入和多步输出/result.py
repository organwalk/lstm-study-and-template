
def success(msg, data):
    return {
        'code': 200,
        'msg': msg,
        'data': data
    }


def error(code, msg):
    return {
        'code': code,
        'msg': msg,
    }


def validate_fields(data, required_fields):
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        if len(missing_fields) == 1:
            msg = f"字段{missing_fields[0]}不能为空"
        else:
            missing_fields_str = "、".join(missing_fields)
            msg = f"字段{missing_fields_str}不能为空"
        return error(500, msg)
    return None
