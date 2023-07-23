from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def test_view_about_model(y, yhat_round_non_negative):
    n_features = y.shape[2]
    fig, axes = plt.subplots(n_features, 1, figsize=(8, 8), sharex=True)
    for i in range(n_features):
        axes[i].plot(y[0, :, i], label='True')
        axes[i].plot(yhat_round_non_negative[0, :, i], label='Prediction')
        axes[i].set_ylabel('Feature ' + str(i + 1))
        axes[i].legend()
    # 添加 x 轴标签
    plt.xlabel('Time')
    # 显示图形
    plt.show()


'''
    2023-07-23 前12小时预测后12小时:
    | MSE: 0.01, R_MSE: 0.04, MAE: 0.01
'''


def evaluate(y, yhat_round_non_negative):
    mse = mean_squared_error(y[0], yhat_round_non_negative[0])
    r_mse = mean_squared_error(y[0], yhat_round_non_negative[0], squared=False)
    mae = mean_absolute_error(y[0], yhat_round_non_negative[0])

    print("MSE: {:.2f}, R_MSE: {:.2f}, MAE: {:.2f}".format(mse, r_mse, mae))

