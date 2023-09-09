from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from evaluation_model import loss


def short_term(x, y, n_steps_in, n_steps_out, n_features):
    model = Sequential()
    model.add(LSTM(500, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(500, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=300, verbose=1)

    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    # history = model.fit(x, y, epochs=300, verbose=1, validation_split=0.2, callbacks=[early_stopping])
    #
    # loss(history)
    model.save('short_term.h5')

    return model


def long_term_with_in_a_week(x, y, n_steps_in, n_steps_out, n_features):
    model = Sequential()
    model.add(LSTM(500, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(500, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=5, verbose=1)

    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    # history = model.fit(x, y, epochs=300, verbose=1, validation_split=0.2, callbacks=[early_stopping])

    return model
