import numpy as np


def get_config(model_type, x, y, evaluation, existing_files, missing_dates, scaler=None):
    config = None
    if model_type == 'ShortTermByLSTM':
        config = {
            'x': x,
            'y': y,
            'scaler': scaler,
            'steps_in': 24,
            'steps_out': 24,
            'features': 8,
            'evaluation': evaluation,
            'date_type': 'H',
            'existing_file': existing_files,
            'missing_dates': missing_dates
        }
    elif model_type == 'LongTermWithinAWeekByLSTM':
        config = {
            'x': x,
            'y': y,
            'steps_in': np.array(x).shape[1],
            'steps_out': np.array(y).shape[1],
            'features': 8,
            'evaluation': evaluation,
            'date_type': 'D',
            'existing_files': existing_files,
            'missing_dates': missing_dates
        }
    return config
