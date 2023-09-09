from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import processing_data
import numpy as np


def start(yhat_inv, date_type):
    real = processing_data.sequence('data/1_data_2023-06-29.csv', 'output', date_type)
    real = np.reshape(real, (1, 24, 8))
    predict = np.reshape(yhat_inv, (1, 24, 8))
    base(real, predict)
    error_curve(real, predict)


def base(real, predict):
    n_features = real.shape[2]
    fig, axes = plt.subplots(n_features, 1, figsize=(8, 8), sharex='all')
    feature_labels = ['Temperature', 'Humidity', 'Speed', 'Direction', 'Rain', 'Sunlight', 'PM2.5', 'PM10']
    for i in range(n_features):
        axes[i].plot(real[0, :, i], label='Real')
        axes[i].plot(predict[0, :, i], label='Prediction')
        axes[i].set_ylabel(feature_labels[i])
        axes[i].legend()
    plt.xlabel('Time')
    plt.show()


def error_curve(real, predict):
    mse = mean_squared_error(real[0], predict[0])
    r_mse = mean_squared_error(real[0], predict[0], squared=False)
    mae = mean_absolute_error(real[0], predict[0])

    print("MSE: {:.2f}, R_MSE: {:.2f}, MAE: {:.2f}".format(mse, r_mse, mae))

    labels = ['MSE', 'R_MSE', 'MAE']
    values = [mse, r_mse, mae]
    plt.bar(labels, values)
    plt.title('Model Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.show()


def loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
