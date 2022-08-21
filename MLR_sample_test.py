from sklearn.svm import SVR
import pandas as pd
import numpy as np
import copy

from sklearn import linear_model
import tools


class Parameters:
    def __init__(self):
        self.input_time_steps = 3
        self.output_time_steps = 1
        self.future_weather_lookahead = 1

        self.file_path = 'data/all_data_with_weather_filled.xlsx'
        self.mask_file_path = 'data/all_data_with_weather_filled_mask.xlsx'
        self.target = ["AQI_S22"]
        self.feature_count = 5
        self.feature_selection_method = "MC"

        # missing value imputation strategy (linear, None)
        self.mvi_strategy = "linear"
        # outlier cleaning
        self.outlier_factor = 1.5
        self.outlier_strategy = None
        self.is_clean_outlier = False
        self.is_exp_smoothing = False

        # experiment parameter
        self.exp_repeat = 30
        self.verbose = 0


def MLR_train_test(params, df, weather_df, mask_df):
    X_train, y_train, X_test, y_test, x_scaler, y_scaler, date_index_train, date_index_test = tools.get_train_test_data(params, df, weather_df, mask_df)

    MLR = linear_model.LinearRegression()
    MLR.fit(X_train, y_train)

    # Svr = SVR(kernel='linear', C=1)
    # Svr.fit(X_train, y_train)

    predicted = MLR.predict(X_test)
    inv_y_test = y_scaler.inverse_transform(y_test)
    inv_y_train = y_scaler.inverse_transform(y_train)
    # make a prediction
    inv_yhat_test = y_scaler.inverse_transform(np.reshape(MLR.predict(X_test), (-1, 1)))
    inv_yhat_train = y_scaler.inverse_transform(np.reshape(MLR.predict(X_train), (-1, 1)))

    # Calculate performance after rescale.
    res = tools.get_report(y_true=inv_y_test, y_pred=inv_yhat_test, doprint=False)
    rmse = res['RMSE']
    rrmse = res["relative_RMSE"]
    smape = res["sMAPE"]
    mape = res["MAPE"]

    y_y_hat = np.concatenate([inv_y_test, inv_yhat_test], axis=1)
    y_y_hat = pd.DataFrame(y_y_hat, columns=["real_y", "predicted_y"], index=date_index_test)
    y_y_hat.to_excel(
        "output/MLR-{}-{}-{}.xlsx".format(params.target[0], params.feature_selection_method, params.is_clean_outlier))

    return y_y_hat, rmse[0], mape[0]


if __name__ == "__main__":
    params = Parameters()

    df, weather_df, mask_df = tools.load_var("data_table_filled.pckl")
    y_y_hat, avg_rmse, avg_mape = MLR_train_test(params, df.copy(), weather_df.copy(), mask_df.copy())

    print('the END')