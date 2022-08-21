import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models import BaseLSTM, BaseAttention
import tensorflow as tf
from itertools import compress
import tools


def model_fit(X_train, X_test, y_train, y_test, params, is_save=False):
    """
    this function fits a LSTM based model on train data

    inputs:
    X_train, X_test: a DataFrame of pollution data
    Y_train, Y_test: a DataFrame of pollution label
    params: configuration parameters. an object of the model_parameters class
    is_save: a boolean flag to save or not to save the trained model
    """
    model = None
    history = None

    if params.model_name == "BaseLSTM":
        # build network
        model = BaseLSTM(hidden_dim=params.hidden_dim,
                         input_shape=(X_train.shape[1], X_train.shape[2]),
                         output_shape=y_train.shape[1],
                         kernel=params.kernel,
                         input_dropout=params.input_dropout,
                         recurrent_dropout=params.recurrent_dropout,
                         stacket_layer_num=params.stacket_layer_num,
                         loss=params.loss)
        history = model.fit(X_train, y_train,
                            epochs=params.epochs,
                            batch_size=params.batch_size,
                            validation_data=(X_test, y_test),
                            verbose=params.verbose,
                            shuffle=False)

        if params.save_model_path and is_save:
            model.save("{}.h5".format(params.save_model_path))

    elif params.model_name == "LSTMAttention":
        model = BaseAttention(hidden_dim=params.hidden_dim,
                              input_shape=(X_train.shape[1], X_train.shape[2]),
                              output_shape=y_train.shape[1],
                              kernel=params.kernel,
                              input_dropout=params.input_dropout,
                              recurrent_dropout=params.recurrent_dropout,
                              stacket_layer_num=params.stacket_layer_num,
                              loss=params.loss)
        history = model.fit(X_train, y_train,
                            epochs=params.epochs,
                            batch_size=params.batch_size,
                            validation_data=(X_test, y_test),
                            verbose=params.verbose,
                            shuffle=False)

        if params.save_model_path and is_save:
            model.save("{}.h5".format(params.save_model_path))
    return model, history


def train_lstm(df, weather_df, mask_df, params):
    """
    this function builds a prediction model based on LSTM network.

    input:
    ------
    df: a dataframe containing the pollution information of all stations
    weather_df: a dataframe containing the weather information
    mask_df: a dataframe which show the location of missing values in df
    params: model configuration parameters. an object of model_parameters class
    """
    if params.is_exp_smoothing:
        df = tools.exponential_smoothing(df)

    features = tools.select_features(df, params.target[0], params.feature_count, params.feature_selection_method)

    if len(features) == 0:
        return

    df = df[features]

    notnull_list = mask_df[params.target].values.squeeze()[params.input_time_steps:]

    # preprocessing step
    if params.is_clean_outlier:
        df = tools.clean_outliers(df, params.outlier_factor)

    data_cols = df.columns
    pollution_cols = data_cols[:]

    x_data, y_data = tools.series_to_supervised(df[pollution_cols], weather_df,
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

    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    # normalize features
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    # num of input signals
    input_dim = int(X_train.shape[1]/params.input_time_steps)
    # num of output signals
    output_dim = int(y_train.shape[1]/params.output_time_steps)

    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], params.input_time_steps, input_dim))
    X_test = X_test.reshape((X_test.shape[0], params.input_time_steps, input_dim))

    model, history = model_fit(X_train, X_test, y_train, y_test, params, is_save=False)

    # make a prediction
    inv_y_test = y_scaler.inverse_transform(y_test)
    inv_yhat_test = y_scaler.inverse_transform(model.predict(X_test))

    # make a prediction
    inv_y_train = y_scaler.inverse_transform(y_train)
    inv_yhat_train = y_scaler.inverse_transform(model.predict(X_train))

    # Calculate performance after rescale.
    res = tools.get_report(y_true=inv_y_test, y_pred=inv_yhat_test)
    rmse = res['RMSE']
    mape = res["MAPE"]

    tf.keras.backend.clear_session()
    del model

    # train lstm model with all data
    totalX = x_data.iloc[:, :].values
    totalY = y_data.iloc[:].values

    X_test = x_data.iloc[-400:, :]
    y_test = y_data.iloc[-400:]

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(totalX)
    totalX = x_scaler.transform(totalX)
    X_test = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit(totalY)
    totalY = y_scaler.transform(totalY)
    y_test = y_scaler.transform(y_test)

    # reshape input to be 3D [samples, timesteps, features]
    totalX = totalX.reshape((totalX.shape[0], params.input_time_steps, input_dim))
    X_test = X_test.reshape((X_test.shape[0], params.input_time_steps, input_dim))

    if params.save_model_path:
        tools.save_var("{}_meta_data.pckl".format(params.save_model_path), [features, x_scaler, y_scaler, params])

    model, history = model_fit(totalX, X_test, totalY, y_test, params, is_save=True)
    tf.keras.backend.clear_session()
    del model

    # End of the training with total data

    print("\n----------------------------------------------")
    print("avg rmse is: {:0.2f}".format(rmse[0]))
    print("avg MAPE is: {:0.2f}".format(mape[0]))

    # inv_y_train = inv_y_train[notnull_list_train]
    # inv_yhat_train = inv_yhat_train[notnull_list_train]
    # date_index_train = list(compress(date_index_train, notnull_list_train))
    # y_y_hat = np.concatenate([inv_y_total, inv_yhat_total], axis=1)

    y_y_hat = np.concatenate([inv_y_test, inv_yhat_test], axis=1)

    real_col = ["real_y_{}".format(i) for i in range(1, params.output_time_steps+1)]
    pred_col = ["{}_y_{}".format(params.model_name, i) for i in range(1, params.output_time_steps+1)]

    cols = real_col + pred_col

    y_y_hat = pd.DataFrame(y_y_hat, columns=cols, index=date_index_test)
    print('the END')
    return y_y_hat