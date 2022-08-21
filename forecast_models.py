import os
import numpy as np
from tensorflow.python.keras.models import load_model
from datetime import timedelta
import tensorflow as tf
from data_helper import *
import tools
import train_stat_models
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


# convert series to supervised learning
def series_to_supervised(pollution_data, weather_data, pollution_in,
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

    poll_weather = concat([pollution_data, weather_agg], axis=1)
    P_cols.extend(tmp_names)
    poll_weather.columns = P_cols

    cols, names = list(), list()
    # input sequence for pollution (t-n, ..., t-1)
    for i in range(pollution_in-1, -1, -1):
        cols.append(poll_weather.shift(i))
        names += [('{}(t-{})'.format(P_cols[j], i)) for j in range(pollution_nvars)]
    x_labels = names[:]

    # forecast sequence (t, t+1, ..., t+n)
    y_labels = []
    for i in range(0, pollution_out):
        cols.append(pollution_data[label_cols].shift(-(i+1)))
        if i == 0:
            tmp_names = [('{}(t)'.format(label_cols[j])) for j in range(label_nvars)]
        else:
            tmp_names = [('{}(t+{})'.format(label_cols[j], i)) for j in range(label_nvars)]
        y_labels += tmp_names
        names += tmp_names
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    x_data = agg[x_labels]
    y_data = agg[y_labels]
    return x_data, y_data


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

    This function prepares data for training fireTS based models
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

    poll_weather = concat([pollution_data, weather_agg], axis=1)
    P_cols.extend(tmp_names)
    poll_weather.columns = P_cols

    x_data = poll_weather[P_cols]
    y_data = poll_weather[label_cols]

    return x_data, y_data


def hybrid_forcast(prediction_df_list, station, pred_step):
    """
    this function loads hybrid prediction model for a specific station and predicts future step.

    input:
    -----
    prediction_df_list: a DataFrame containing the prediction of other prediction models
    station: specifies the target station for prediction
    pred_step: specifies how far we want to predict

    return:
    ------
    a vector containing the prediction resutls
    """
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    hybrid_model_path = dir_path + "models/{}_{}_{}.pckl".format("hybrid_svr", station, pred_step)
    hybrid_model_path_meta = dir_path + "models/{}_{}_{}_meta_data.pckl".format("hybrid_svr", station, pred_step)
    hybrid_model = tools.load_var(hybrid_model_path)
    feature_label = tools.load_var(hybrid_model_path_meta)[0]

    hybrid_model_input = prediction_df_list[feature_label]
    hybrid_model_input_date = hybrid_model_input.index.tolist()

    y_hat = hybrid_model.predict(hybrid_model_input.values)
    return y_hat


class ForecastModels:
    """
    this is a class to load pretrained models and use them to predict future AQI
    """
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
        # self.model_list = model_list
        # self.station_list = station_list
        # self.model_array = self.__load_models()

    def __load_lstm_models(self, station, model_name):
        """
        load a pre-trained LSTM model and do the prediction

        input:
        ------
        station: specifies the station that we what to predict its future AQI
        model_name: LSTM model name

        return:
        -------
        returns the loaded LSTM model and its meta data
        """
        tmp_meta_path = self.dir_path + "models/{}_{}_meta_data.pckl".format(model_name, station)
        if not os.path.isfile(tmp_meta_path):
            print("this model is not exist..")
            return [None, None, None, None]

        meta = tools.load_var(tmp_meta_path)

        model_graph = tf.Graph()
        with model_graph.as_default():
            model_session = tf.compat.v1.Session(graph=model_graph)
            with model_session.as_default():
                # load model
                model = load_model(self.dir_path + 'models/{}_{}.h5'.format(model_name, station))
                # summarize model.
                # model.summary()
                print("{}_{} model loaded...".format(model_name, station))
        return model_graph, model_session, model, meta

    def __load_fireTS_models(self, station, model_name, pred_step):
        """
        load a pre-trained fireTS model and do the prediction

        input:
        ------
        station: specifies the station that we what to predict its future AQI
        model_name: fireTS model name
        pred_step: specifies how far we want to predict

        return:
        -------
        returns the loaded fireTS model and its meta data
        """
        tmp_meta_path = self.dir_path + "models/{}_{}_{}_meta_data.pckl".format(model_name, station, pred_step)
        if not os.path.isfile(tmp_meta_path):
            print("this model is not exist..")
            return [None, None]

        meta = tools.load_var(tmp_meta_path)
        model = tools.load_var(self.dir_path + 'models/{}_{}_{}.pckl'.format(model_name, station, pred_step))
        print("{}_{} model loaded...".format(model_name, station))
        return model, meta

    def __load_statistical_models(self, station, model_name):
        """
        load a pre-trained statistical model and do the prediction

        input:
        ------
        station: specifies the station that we what to predict its future AQI
        model_name: statistical model name

        return:
        -------
        returns the loaded statistical model and its meta data
        """
        tmp_meta_path = self.dir_path + "models/{}_{}_meta_data.pckl".format(model_name, station)
        if not os.path.isfile(tmp_meta_path):
            print("this model is not exist..")
            return [None, None]

        meta = tools.load_var(tmp_meta_path)
        model = tools.load_var(self.dir_path + 'models/{}_{}.pckl'.format(model_name, station))
        print("{}_{} model loaded...".format(model_name, station))
        return model, meta

    def predict(self, station, model_name, pred_step, poll_df, weather_df):
        """
        this function loads a specified pre-trained model and then do the future prediction based on observed
        prev values

        input:
        ------
        station: the station Id that we want to predict AQI for
        model_name: the pre-trained model name to use for prediction
        pred_step: specifies how far we want to predict
        poll_df: A DataFrame of different observed pollution features
        weather_df: A DataFrame of different observed weather features

        return:
        ------
        A DataFrame named y_y_hat, which contains real values and prediction values in different days
        """
        if model_name in ["BaseLSTM", "LSTMAttention"]:
            model_graph, model_session, model, meta = self.__load_lstm_models(station, model_name)

            # input_time_steps = 3
            if model_graph is None:
                return []

            features, x_scaler, y_scaler, params = meta

            cols = set(poll_df.columns)
            not_in_cols = set(features) - cols
            for t in not_in_cols:
                poll_df.insert(len(cols), t, np.zeros(poll_df.shape[0]), True)

            poll_df = poll_df[features]
            # poll_df = missing_value_imputation(poll_df)

            x_data, y_data = series_to_supervised(poll_df, weather_df,
                                                  pollution_in=params.input_time_steps,
                                                  pollution_out=params.output_time_steps,
                                                  weather_in=params.future_weather_lookahead,
                                                  label_cols=params.target, dropnan=False)
            y_data = y_data.fillna(0)

            date_index_test = poll_df.index.tolist()
            date_index_test = date_index_test[params.input_time_steps-1:]
            for t in range(len(date_index_test)):
                date_index_test[t] = date_index_test[t] + timedelta(days=1)

            X_test = x_data.values[params.input_time_steps-1:, :]
            y_test = y_data.values[params.input_time_steps-1:]

            X_test = x_scaler.transform(X_test)

            input_dim = int(X_test.shape[1] / params.input_time_steps)
            X_test = X_test.reshape((X_test.shape[0], params.input_time_steps, input_dim))

            with model_graph.as_default():
                with model_session.as_default():
                    y_hat = model.predict(X_test)

            inv_yhat = y_scaler.inverse_transform(y_hat)

            y_y_hat = np.concatenate([y_test, inv_yhat], axis=1)
            real_col = ["real_y_{}".format(i) for i in range(1, params.output_time_steps + 1)]
            pred_col = ["{}_y_{}".format(params.model_name, i) for i in range(1, params.output_time_steps + 1)]
            cols = real_col + pred_col
            y_y_hat = pd.DataFrame(y_y_hat, columns=cols, index=date_index_test)

        elif model_name in ["fireTSaR", "fireTSaX", "fireTSnR", "fireTSnX"]:
            all_res = []
            for step in range(1, pred_step+1):
                model, meta = self.__load_fireTS_models(station, model_name, step)

                if model is None:
                    return []

                features, x_scaler, y_scaler, params = meta

                cols = set(poll_df.columns)
                not_in_cols = set(features) - cols
                for t in not_in_cols:
                    poll_df.insert(len(cols), t, np.zeros(poll_df.shape[0]), True)

                poll_df = poll_df[features]
                # poll_df = missing_value_imputation(poll_df)

                x_data, y_data = fireTS_series_to_supervised(poll_df, weather_df,
                                                             pollution_in=params.input_time_steps,
                                                             pollution_out=step,
                                                             weather_in=params.future_weather_lookahead,
                                                             label_cols=params.target, dropnan=True)
                date_index_test = x_data.index.tolist()
                for i in range(step):
                    date_index_test.append(date_index_test[-1] + timedelta(days=1))
                X_test = x_data.values
                X_test = np.concatenate((X_test, np.zeros([step, X_test.shape[1]])))

                y_test = y_data.values
                y_test = np.concatenate((y_test, np.zeros([step, y_test.shape[1]])))

                X_test = x_scaler.transform(X_test)
                y_test = y_scaler.transform(y_test)

                if params.firets_model_name == "autoregression":
                    ypred_test = model.predict(X_test, y_test.ravel())
                else:
                    ypred_test = model.predict(X_test, y_test.ravel(), step=step)

                nan_count = params.input_time_steps + step - 1

                date_index_test = date_index_test[nan_count:]
                y_test = y_test[nan_count:]
                ypred_test = ypred_test[nan_count:]
                ypred_test = np.expand_dims(ypred_test, axis=1)

                y_test = y_scaler.inverse_transform(y_test)
                inv_yhat = y_scaler.inverse_transform(ypred_test)

                y_y_hat = np.concatenate([y_test, inv_yhat], axis=1)
                y_y_hat = pd.DataFrame(y_y_hat, columns=["real_y_{}".format(step),
                                                         "{}_y_{}".format(params.model_name, step)],
                                       index=date_index_test)
                all_res.append(y_y_hat)

            all_res = pd.concat(all_res, axis=1)
            for i in range(pred_step):
                cols = all_res.columns
                shift_cols = []
                for c in cols:
                    if c.endswith("_{}".format(i + 1)):
                        shift_cols.append(c)
                all_res[shift_cols] = all_res[shift_cols].shift(periods=-i)
            y_y_hat = all_res

        else:
            model, meta = self.__load_statistical_models(station, model_name)
            if model is None:
                return []

            params, history, exog = meta

            poll_df = poll_df[params.target[0]]
            date_index_test = poll_df.index.tolist()

            history_index = sorted(list(set(history.index.tolist()) - set(date_index_test)))
            history = history.loc[history_index]
            history.columns = ['y']

            test = poll_df.copy()
            test.columns = ['y']
            test.name = 'y'

            if exog is not None:
                exog_test = weather_df[params.exog_lst].copy()
                exog_index = sorted(list(set(exog.index.tolist()) - set(date_index_test)))
                exog = exog.loc[exog_index]

                exog.columns = params.exog_lst
                exog_test.columns = params.exog_lst

                # concat dataframes
                history = pd.concat([history, exog], axis=1)
                test = pd.concat([test, exog_test], axis=1)

            # log-transform
            test = np.log1p(test)

            if isinstance(test, pd.Series):
                test = test.to_frame()
            if isinstance(history, pd.Series):
                history = history.to_frame()

            model, preds, tsts = train_stat_models.model_fit(X_train=history.copy(), X_test=test.copy(),
                                                             params=params, is_save=False)
            date_index_test.append(date_index_test[-1] + timedelta(days=1))

            if params.model_name == "ARIMA":
                output = model.forecast(steps=params.output_time_steps)[0]
                yhat = output.tolist()
            elif params.model_name == "ARIMAXW" or params.model_name == "ARIMAXH":
                exog_var_forcasted = np.ones(pred_step) * test[params.exog_lst].values[-1][0]
                # exog_var_forcasted = test[params.exog_lst].values[-1][0]
                output = model.forecast(steps=params.output_time_steps, exog=exog_var_forcasted)[0]
                yhat = output.tolist()
            else:
                history_tmp = history.copy().values
                test = test.copy().values
                history_tmp = np.concatenate([history_tmp, test], axis=0).ravel().tolist()

                yhat = []
                for inc in range(pred_step):
                    model = SimpleExpSmoothing(history_tmp)
                    model = model.fit(smoothing_level=None)
                    pred_val = model.forecast()[0]
                    history_tmp.append(pred_val)
                    yhat.append(pred_val)

            preds.append(yhat)
            tsts.append([0 for i in range(pred_step)])
            preds = np.expm1(np.asarray(preds))
            tsts = np.expm1(np.asarray(tsts))


            y_y_hat = np.concatenate([tsts, preds], axis=1)

            real_col = ["real_y_{}".format(i) for i in range(1, params.output_time_steps + 1)]
            pred_col = ["{}_y_{}".format(params.model_name, i) for i in range(1, params.output_time_steps + 1)]

            cols = real_col + pred_col

            y_y_hat = pd.DataFrame(y_y_hat, columns=cols, index=date_index_test)
        return y_y_hat
