def validate_fields(user_req_json, server_required_fields):
    missing_fields = [field for field in server_required_fields if not user_req_json.get(field)]
    if missing_fields:
        if len(missing_fields) == 1:
            msg = f"字段{missing_fields[0]}不能为空"
        else:
            missing_fields_str = "、".join(missing_fields)
            msg = f"字段{missing_fields_str}不能为空"
        return msg
    return None
