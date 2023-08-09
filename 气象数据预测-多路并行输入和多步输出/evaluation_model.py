from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def test_view_about_model(y, yhat):
    n_features = y.shape[2]
    fig, axes = plt.subplots(n_features, 1, figsize=(8, 8), sharex=True)
    feature_labels = ['Temperature', 'Humidity', 'Speed', 'Direction', 'Rain', 'Sunlight', 'PM2.5', 'PM10']
    for i in range(n_features):
        axes[i].plot(y[0, :, i], label='True')
        axes[i].plot(yhat[0, :, i], label='Prediction')
        axes[i].set_ylabel(feature_labels[i])
        axes[i].legend()
    # 添加 x 轴标签
    plt.xlabel('Time')
    # 显示图形
    plt.show()


def evaluate(real, predict):
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

