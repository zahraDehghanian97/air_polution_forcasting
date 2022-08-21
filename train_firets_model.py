from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

import tools
from tools import *
import copy

from fireTS.models import NARX, DirectAutoRegressor
from itertools import compress

import model_parameters


# convert series to supervised learning
def fireTS_series_to_supervised(pollution_data, weather_data, pollution_in,
                                pollution_out, weather_in, dropnan, label_cols=None):
    """
    this function gets input data as time series DataFrame and builds data row feature vectors
        for training a model.
    pollution_data: a DataFrame which contains the history of pollution information. The DataFrame
        index is in datetime format
    weather_data: a DataFrame which contains the history of weather information like pollution_data
    pollution_in: represents the input time steps. when its values is n we consider n previous time
        step information to predict future
    pollution_out: represents the output time steps to predict
    weather_in: shows the future time steps of weather information that we can use to predict pollution.
        the value of 1 only uses the current time weather information. the value 2 uses the informations
        of steps t and t+1 and so on.
    label_cols: specifies the target feature that we want to predict. Its a list of string which are
        the names of the target columns in input DataFrame "pollution_data"
    dropnan: this is a boolian flag. if it's True the raws in data which contain nan value will be removed
    """

    if label_cols is None:
        label_cols = pollution_data.columns
    pollution_nvars = 1 if type(pollution_data) is list else pollution_data.shape[1]
    weather_nvars = 1 if type(weather_data) is list else weather_data.shape[1]
    pollution_nvars = pollution_nvars + weather_in * weather_nvars

    label_nvars = len(label_cols)
    P_cols = list(pollution_data.columns.values)
    W_cols = weather_data.columns

    tmp_cols = []
    tmp_names = []
    for i in range(0, weather_in):
        tmp_cols.append(weather_data.shift(-i))
        if i == 0:
            tmp_names += [('{}(t)'.format(W_cols[j])) for j in range(weather_nvars)]
        else:
            tmp_names += [('{}(t+{})'.format(W_cols[j], i)) for j in range(weather_nvars)]

    weather_agg = concat(tmp_cols, axis=1)
    weather_agg.columns = tmp_names
    weather_agg = weather_agg.fillna(method='ffill')

    poll_weather = pollution_data.copy()
    poll_weather = poll_weather.merge(weather_agg, how='left', left_index=True, right_index=True)
    # poll_weather = concat([pollution_data, weather_agg], axis=1, join="inner")

    P_cols.extend(tmp_names)
    poll_weather.columns = P_cols

    x_data = poll_weather[P_cols]
    y_data = poll_weather[label_cols]

    return x_data, y_data


def model_fit(X_train, X_test, y_train, y_test, pred_step, params, is_save):
    """
    this function fits a fireTS model on train data

    inputs:
    X_train, X_test: a DataFrame of pollution data
    Y_train, Y_test: a DataFrame of pollution label
    pred_step: specifies how far we want to do the prediction
    params: configuration parameters. an object of the model_parameters class
    is_save: a boolean flag to save or not to save the trained model
    """
    # build network
    if params.firets_model_name == "autoregression":
        model = DirectAutoRegressor(params.base_regressor, auto_order=params.input_time_steps,
                                    exog_order=[params.input_time_steps for i in range(X_train.shape[1])],
                                    exog_delay=[0 for i in range(X_train.shape[1])], pred_step=pred_step)
        model.fit(X_train, y_train.ravel())
    else:
        model = NARX(params.base_regressor, auto_order=params.input_time_steps,
                     exog_order=[params.input_time_steps for i in range(X_train.shape[1])],
                     exog_delay=[0 for i in range(X_train.shape[1])])

        model.fit(X_train, y_train.ravel())

    if params.save_model_path and is_save:
        tools.save_var("{}_{}.pckl".format(params.save_model_path, pred_step), model)

    return model


def train_fireTS(df, weather_df, mask_df, params):
    """
    this function builds a prediction model based on fireTS.

    input:
    ------
    df: a dataframe containing the pollution information of all stations
    weather_df: a dataframe containing the weather information
    mask_df: a dataframe which show the location of missing values in df
    params: model configuration parameters. an object of model_parameters class
    """
    if params.is_exp_smoothing:
        df = exponential_smoothing(df)

    features = select_features(df, params.target[0], params.feature_count, params.feature_selection_method)

    if len(features) == 0:
        return

    df = df[features]

    notnull_list = mask_df[params.target].values.squeeze()[params.input_time_steps:]

    # preprocessing step
    if params.is_clean_outlier:
        df = clean_outliers(df, params.outlier_factor)

    data_cols = df.columns
    pollution_cols = data_cols[:]

    x_data, y_data = fireTS_series_to_supervised(df[pollution_cols], weather_df,
                                                 pollution_in=params.input_time_steps,
                                                 pollution_out=params.output_time_steps,
                                                 weather_in=params.future_weather_lookahead,
                                                 label_cols=params.target, dropnan=True)

    X_train = x_data.iloc[:-400, :]
    y_train = y_data.iloc[:-400]
    notnull_list_train = np.array(notnull_list[:-400], dtype=bool)

    X_test = x_data.iloc[-400:, :]
    y_test = y_data.iloc[-400:]
    notnull_list_test = np.array(notnull_list[-400:], dtype=bool)

    X_test = X_test.loc[notnull_list_test, :]
    y_test = y_test.loc[notnull_list_test]

    if X_test.shape[0] == 0 or X_train.shape[0] == 0:
        return None

    date_index_train = X_train.index.tolist()
    date_index_test = X_test.index.tolist()

    # normalize features
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    all_res = []
    for pred_step in range(1, params.output_time_steps+1):

        # training the model
        model = model_fit(X_train, X_test, y_train, y_test, pred_step, params, is_save=False)

        inv_y_test = y_scaler.inverse_transform(y_test)
        inv_y_train = y_scaler.inverse_transform(y_train)

        # make a prediction
        if params.firets_model_name == "autoregression":
            ypred_test = model.predict(X_test, y_test.ravel())
            ypred_train = model.predict(X_train, y_train.ravel())
        else:
            ypred_test = model.predict(X_test, y_test.ravel(), step=pred_step)
            ypred_train = model.predict(X_train, y_train.ravel(), step=pred_step)

        nan_count = params.input_time_steps + pred_step - 1

        ypred_test = ypred_test[nan_count:]
        ypred_test = np.expand_dims(ypred_test, axis=1)

        ypred_train = ypred_train[nan_count:]
        ypred_train = np.expand_dims(ypred_train, axis=1)

        inv_yhat_test = y_scaler.inverse_transform(ypred_test)
        inv_yhat_train = y_scaler.inverse_transform(ypred_train)

        inv_y_test = inv_y_test[nan_count:]
        inv_y_train = inv_y_train[nan_count:]

        date_index_train_tmp = date_index_train[nan_count:]
        date_index_test_tmp = date_index_test[nan_count:]

        # Calculate performance afte r rescale.
        res = get_report(y_true=inv_y_test, y_pred=inv_yhat_test, doprint=False)
        rmse = res['RMSE']
        mape = res["MAPE"]

        del model

        # train fireTs model with all data
        totalX = x_data.iloc[:, :].values
        totalY = y_data.iloc[:].values

        X_test_tmp = x_data.iloc[-400:, :]
        y_test_tmp = y_data.iloc[-400:]

        x_scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaler.fit(totalX)
        totalX = x_scaler.transform(totalX)
        X_test_tmp = x_scaler.transform(X_test_tmp)

        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler.fit(totalY)
        totalY = y_scaler.transform(totalY)
        y_test_tmp = y_scaler.transform(y_test_tmp)

        if params.save_model_path:
            tools.save_var("{}_{}_meta_data.pckl".format(params.save_model_path, pred_step), [features, x_scaler, y_scaler, params])

        model = model_fit(totalX, X_test_tmp, totalY, y_test_tmp, pred_step, params, is_save=True)
        # End of the training with total data

        print("\n----------------------------------------------")
        print("avg rmse is: {:0.2f}".format(rmse[0]))
        print("avg MAPE is: {:0.2f}".format(mape[0]))

        # inv_y_train = inv_y_train[notnull_list_train]
        # inv_yhat_train = inv_yhat_train[notnull_list_train]
        # date_index_train = list(compress(date_index_train, notnull_list_train))
        # y_y_hat = np.concatenate([inv_y_train, inv_yhat_train], axis=1).transpose((1, 0))

        y_y_hat = np.concatenate([inv_y_test, inv_yhat_test], axis=1)
        y_y_hat = pd.DataFrame(y_y_hat, columns=["real_y_{}".format(pred_step), "{}_y_{}".format(params.model_name, pred_step)],
                               index=date_index_test_tmp)
        all_res.append(y_y_hat)
    all_res = pd.concat(all_res, axis=1)
    for i in range(params.output_time_steps):
        cols = all_res.columns
        shift_cols = []
        for c in cols:
            if c.endswith("_{}".format(i+1)):
                shift_cols.append(c)
        all_res[shift_cols] = all_res[shift_cols].shift(periods=-i)
    return all_res
