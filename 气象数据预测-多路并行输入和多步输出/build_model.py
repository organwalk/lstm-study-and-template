from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


def training_short_term_model_about_oneday(x, y, n_steps_in, n_steps_out, n_features):
    # print(x.shape)  # (2, 24, 8)
    # print(y.shape)  # (2, 24, 8)
    model = Sequential()

    model.add(LSTM(500, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(500, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))

    model.compile(optimizer='adam', loss='mse')

    model.fit(x, y, epochs=500, verbose=1)

    return model
