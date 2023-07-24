import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import load_model
import evaluation_model
import pre_processing_data

# x, y, n_features, n_steps_in, n_steps_out, scaler_out = pre_processing_data.get_data_other_day()
x, y, n_features, n_steps_in, n_steps_out, scaler_out = pre_processing_data.get_data_same_day()

# model = Sequential()
# model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
# model.add(RepeatVector(n_steps_out))
# model.add(LSTM(200, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(n_features)))
# model.compile(optimizer='adam', loss='mse')
#
# model.fit(x, y, epochs=3000, verbose=0)

model = load_model('model_file_v0.1.h5')

x_input = x
x_input = x_input.reshape((1, n_steps_in, n_features))

yhat = model.predict(x_input, verbose=0)
yhat_round = np.round(yhat).astype(int)
yhat_round_non_negative = np.clip(yhat_round, a_min=0, a_max=None)
yhat_round_non_negative = yhat_round_non_negative.reshape((n_steps_out, n_features))

yhat_inv = scaler_out.inverse_transform(yhat_round_non_negative)
yhat_inv = yhat_inv.reshape((n_steps_out, n_features))
yhat_inv = np.round(yhat_inv).astype(int)

yhat_inv_list = [list(row) for row in yhat_inv]

print(yhat_inv_list)

# 评估模型
evaluation_model.test_view_about_model(y, yhat)
evaluation_model.evaluate(y, yhat)

# model.save('model_file_v0.1.h5')
