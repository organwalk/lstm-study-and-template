import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import evaluation_model
import pre_processing_data

x, y, n_features, n_steps_in, n_steps_out = pre_processing_data.get_data()

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

model.fit(x, y, epochs=3000, verbose=1)

x_input = x
x_input = x_input.reshape((1, n_steps_in, n_features))

yhat = model.predict(x_input, verbose=1)
yhat_round = np.round(yhat).astype(int)
yhat_round_non_negative = np.clip(yhat_round, a_min=0, a_max=None)
print([list(row) for row in yhat_round_non_negative[0]])


# 评估模型
evaluation_model.test_view_about_model(y, yhat_round_non_negative)
evaluation_model.evaluate(y, yhat_round_non_negative)
