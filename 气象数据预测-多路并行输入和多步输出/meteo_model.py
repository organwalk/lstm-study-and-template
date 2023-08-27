import processing_data
from config import get_config
import loading_model


def start_predict(json):
    model_type = json.get('model_type')
    result = None
    if model_type == 'ShortTermByLSTM':
        x, y, scaler, existing_file, missing_dates = processing_data.short_term_pre_process(json)
        print(existing_file)
        result = loading_model.short_term_model(
            get_config(model_type, x, y, False, existing_file, missing_dates, scaler)
        )
    elif model_type == 'LongTermWithinAWeekByLSTM':
        # 此处后期应该改为(7, 7, json)
        x, y, existing_files, missing_dates = processing_data.long_term_with_in_a_week_pre_process(2, 1, json)
        result = loading_model.long_term_with_in_a_week_model(
            get_config(model_type, x, y, False, existing_files, missing_dates)
        )
    return result
