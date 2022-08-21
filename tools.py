import pickle
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import os
from pandas import concat
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from converter import HolidayCheck
import scipy.stats as stats
import geopy.distance


def load_var(load_path):
    """
    this function loads variables in pickle format from disk
    """
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    """
    this function saves variables in pickle format to disk
    """
    print("saving vars ...")
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    print("variable saved.")
    file.close()


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
    for i in range(pollution_in, 0, -1):
        cols.append(poll_weather.shift(i))
        names += [('{}(t-{})'.format(P_cols[j], i)) for j in range(pollution_nvars)]
    x_labels = names[:]

    # forecast sequence (t, t+1, ..., t+n)
    y_labels = []
    for i in range(0, pollution_out):
        cols.append(pollution_data[label_cols].shift(-i))
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


def add_holiday_features(input_df):
    """
    this function builds a feature vector in the form of [is_weekend, is_holiday, day_of_the_week, month_of_the_year]
    for each row data
    """
    holiday_check = HolidayCheck()
    date_list = input_df.index.tolist()
    features = []
    for tmp_date in date_list:
        #  export the day feature in order [is_weekend, is_holiday, day_of_the_week, month_of_the_year]
        tmp_features = holiday_check.get_holiday_status_of_datetime(tmp_date)
        features.append(tmp_features)
    holiday_df = pd.DataFrame(features, index=date_list,
                              columns=["is_weekend", "is_holiday", "day_of_the_week", "month_of_the_year"])
    output_df = pd.concat([input_df, holiday_df], axis=1)
    return output_df


def generate_batches(x, y, batch_size):
    """
    this function build batches from train data to use in train step

    input:
    ------
    x: data matrix
    y: data label
    batch_size: size of each batch
    """
    x_len = x.shape[0]
    num_batches_per_epoch = int(x_len/batch_size)

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, x_len)
        tmp_x = x[start_index:end_index, :].copy()
        tmp_y = y[start_index:end_index, :].copy()
        yield (tmp_x, tmp_y)


# create a differenced series
def difference(dataset):
    return dataset.diff()


# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob


def my_mvi(X_missing, method, imp=None):
    """
    this function is used to fill missing values based on desired method
    """
    if method in ["BayesianRidge", "DecisionTreeRegressor",
                  "ExtraTreesRegressor", "KNeighborsRegressor"]:

        predicted = imp.transform(X_missing.values)
    elif method in ["linear", "cubic"]:
        predicted = X_missing.interpolate(method=method).values
    elif method in ["ffill", "bfill"]:
        predicted = X_missing.fillna(method=method).values
    else:
        predicted = X_missing.fillna(X_missing.rolling(5, min_periods=1, center=True).mean()).values

    return predicted


def fill_data_frame(data_df):
    """
    this function fills missing values in a dataframe

    input:
    ------
    dataframe of data

    return:
    ------
    a filled dataframe
    """
    tags = data_df.columns.tolist()
    for i, t in enumerate(tags):
        if i % 10 == 0:
            print("filling missing values of feature {}: {}/{}".format(t, i, len(tags)))
        data_df = fill_column(data_df, t)

    if data_df.isnull().values.any():
        data_df = data_df.interpolate(method='linear', limit_direction='both')
        data_df = data_df.fillna(0)

    return data_df


def fill_column(data_df, target, strategy=None, dropnan=True):
    """
    this function fills missing values for specific column in a dataframe with pre-trained models
    """
    final_df = data_df.copy()

    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    imputer_model_path = "{}{}.pckl".format(dir_path + "imputer/", target)
    if os.path.isfile(imputer_model_path):
        imputer_list, correlated_tags = load_var(imputer_model_path)
    else:
        return final_df

    selected_df = final_df[correlated_tags].copy()
    tags = selected_df.columns
    target_index = np.where(tags.values == target)
    target_index = target_index[0][0]

    for el in imputer_list:
        method, imputer = el
        predicted = my_mvi(selected_df.copy(), method, imputer)
        predicted = predicted[:, target_index]

        null_idx = final_df[target].isnull().values.squeeze()
        final_df[target].loc[null_idx] = predicted[null_idx]

        tmp_df = final_df[target].copy()
        if tmp_df.isnull().values.any():
            continue
        else:
            break
    return final_df


def clean_outliers(data_df, outlier_factor, strategy=None):
    """
    this function finds outliers in a row data and replace it with a maximum or minimum value
    """
    columns = data_df.columns
    Q1 = data_df.quantile(0.25)
    Q3 = data_df.quantile(0.75)
    IQR = Q3 - Q1

    lb = (Q1 - outlier_factor * IQR)
    ub = (Q3 + outlier_factor * IQR)
    for col in columns:
        data_df[col].loc[data_df[col] > ub[col]] = ub[col]
        data_df[col].loc[data_df[col] < lb[col]] = lb[col]
    return data_df


def mutual_information_cal(data_df, t_list):
    """
    this function estimates mutual information of two vector
    """
    mutual_inf_list = []
    for i in range(len(t_list)):
        t1 = t_list[i]
        row_list = [str(0)]
        for k in range(i):
            row_list.append(str(0))
        for j in range(i+1, len(t_list)):
            t2 = t_list[j]
            tmp = data_df[[t1, t2]].copy()
            tmp.dropna(inplace=True)
            tmp1 = tmp[[t1]].values
            tmp2 = np.squeeze(tmp[[t2]].values)
            try:
                tmp_mi = mutual_info_regression(tmp1, tmp2)
            except:
                tmp_mi = [0]
            row_list.append("{:0.2f}".format(tmp_mi[0]))
        mutual_inf_list.append(row_list)
    return mutual_inf_list


def select_features(df, target, feature_count, strategy):
    """
    this function selects features based on some measure to train prediction models
    input:
    ------
    df: dataframe containing AQI information of all stations
    feature_count: number of the desired features to select
    strategy: feature selection strategy
    """
    distance_th = 10  # km
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    station_locations = pd.read_excel(dir_path + "station_id.xlsx")
    station_distances = {}
    for i in range(station_locations.shape[0]):
        for j in range(i, station_locations.shape[0]):
            station1_id = station_locations.iloc[i]["id"]
            station1_lat = station_locations.iloc[i]['Latitude']
            station1_long = station_locations.iloc[i]['Longitude']

            station2_id = station_locations.iloc[j]["id"]
            station2_lat = station_locations.iloc[j]['Latitude']
            station2_long = station_locations.iloc[j]['Longitude']

            coords_1 = (station1_lat, station1_long)
            coords_2 = (station2_lat, station2_long)

            dist = geopy.distance.geodesic(coords_1, coords_2).km

            if station1_id == 1000 or station2_id == 1000:
                dist = 0

            station_distances[(station1_id, station2_id)] = dist
            station_distances[(station2_id, station1_id)] = dist

    # selecting the most correlated features
    # strategy = MC (Most Correlated) or MMI(Most Mutual Information)
    tags = df.columns
    corrMatrix = df.corr().abs()

    cors = []
    mi_list = []
    target_station = int(target.split("_")[-1][1:])
    for t in tags:
        t_station = int(t.split("_")[-1][1:])

        if target.startswith("AQI") and not t.startswith("AQI"):
            target_endpart = target.split("_")[-1]
            t_endpart = t.split("_")[-1]
            if target_endpart != t_endpart:
                continue

        if station_distances[(target_station, t_station)] <= distance_th:
            tmp_corr = corrMatrix[target][t]
            if np.isnan(tmp_corr):
                continue
            else:
                cors.append((tmp_corr, t))

            if t != target:
                tmp = df[[t, target]].copy()
                tmp.dropna(inplace=True)
                tmp1 = tmp[[t]].values
                tmp2 = np.squeeze(tmp[[target]].values)

                if tmp1.shape[0] < 3:
                    continue
                try:
                    tmp_mi = mutual_info_regression(tmp1, tmp2)
                except:
                    tmp_mi = 0
                mi_list.append((tmp_mi, t))
        else:
            pr = "stations are far apart"

    if strategy == "MC":
        corr_list = sorted(cors, key=lambda x: x[0], reverse=True)
        corr_list = corr_list[:feature_count]
        correlated_tags = [t[1] for t in corr_list]
        corrMatrix = corrMatrix.loc[correlated_tags, correlated_tags]
        corr_list = []
        for t1 in correlated_tags:
            tmp_list = []
            for t2 in correlated_tags:
                tmp_corr = corrMatrix[t1][t2]
                tmp_list.append("{:0.2}".format(tmp_corr))
            corr_list.append(tmp_list)
        # draw_table(correlated_tags, correlated_tags, corr_list, "correlation matrix for target {}".format(target))
        return correlated_tags
    else:
        mi_list = sorted(mi_list, key=lambda x: x[0], reverse=True)
        mi_list = mi_list[:feature_count]
        mi_tags = [t[1] for t in mi_list]
        mi_tags.append(target)
        mutual_inf_list = mutual_information_cal(df, mi_tags)
        # draw_table(mi_tags, mi_tags, mutual_inf_list, "Mutual Information for target {}".format(target))
        # print("mi values is: ", mi_tags)
        return mi_tags


def simple_exp_smoothing(data):
    fited_data = SimpleExpSmoothing(data).fit()
    return fited_data.fittedvalues


def exponential_smoothing(df):
    cols = df.columns
    for col in cols:
        es_df = simple_exp_smoothing(df[[col]])
        # if col == "PM10_S14":
        #     get_plot(df[col], es_df, "tmp.png", "actual vs smooth version",
        #              y_pred_label="exp smooth", y_label="Pollution amount")
        df[col].iloc[:-400] = es_df.iloc[:-400]
    return df


def data_loader(file_path, mask_path=None):
    """
    this function loads pollution data from file to train prediction models
    """
    df = pd.read_excel(file_path)
    df.drop_duplicates(inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['date'], inplace=True)
    df = df.set_index('date')

    weather_cols = ["WindSpeed10", "WindDegree10", "Temperture",
                    "RelativeHumidity", "DewpointTemperature"]
    weather_df = df[weather_cols]
    df = df.drop(weather_cols, axis=1)
    df = df.drop(['Pressure', 'Visibility'], axis=1)

    for t in df.columns:
        df[t] = pd.to_numeric(df[t], errors='coerce')

    mask_df = None
    if mask_path:
        mask_df = pd.read_excel(mask_path)
        mask_df.drop_duplicates(inplace=True)

        mask_df['date'] = pd.to_datetime(mask_df['date'])
        mask_df.sort_values(['date'], inplace=True)
        mask_df = mask_df.set_index('date')

        mask_df = mask_df.drop(weather_cols, axis=1)

        cols = mask_df.columns
        mask_df[cols] = mask_df[cols].astype('bool')
    return df, weather_df, mask_df


def get_train_test_data(params, df, weather_df, mask_df):
    """
    This function splits pollution data into train set and test set
    """
    if params.is_exp_smoothing:
        df = exponential_smoothing(df)

    features = select_features(df, params.target[0], params.feature_count,
                               params.feature_selection_method)
    df = df[features]

    notnull_list = mask_df[params.target].values.squeeze()[params.input_time_steps:]

    # preprocessing step
    if params.is_clean_outlier:
        df = clean_outliers(df, params.outlier_factor)

    # df = missing_value_imputation(df, target=params.target, strategy=params.mvi_strategy, dropnan=False)

    data_cols = df.columns
    pollution_cols = data_cols[:]

    x_data, y_data = series_to_supervised(df[pollution_cols], weather_df,
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

    return X_train, y_train, X_test, y_test, x_scaler, y_scaler, date_index_train, date_index_test


def draw_table(rowLabels, colLabels, content, title, colWidths=None):
    """
    this function draws a table of test results in the output
    """
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_axis_off()
    table = ax.table(
        cellText=content,
        rowLabels=rowLabels,
        colLabels=colLabels,
        rowColours=["palegreen"] * len(rowLabels),
        colColours=["palegreen"] * len(colLabels),
        cellLoc='center',
        loc='upper center',
        colWidths=colWidths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    # table.scale(2, 2)

    ax.set_title(title, fontweight="bold")
    plt.show()


def get_report(y_true, y_pred, doprint=True, test_cols=None):
    """
    this function is used to evaluate the performance of the prediction models
    """
    try:
        output_step = y_pred.shape[1]
    except:
        output_step = 1

    output_step = 1

    step_error = []
    for i in range(output_step):
        y_inv = y_true[:, i]
        yhat_inv = y_pred[:, i]

        notnan_index = ~np.isnan(y_inv)

        y_inv = y_inv[notnan_index]
        yhat_inv = yhat_inv[notnan_index]

        rmse = RMSE(y_inv, yhat_inv)
        relative_rmse = RRMSE(y_inv, yhat_inv)
        smape = sMAPE(y_inv, yhat_inv)
        mape = MAPE(y_true=y_inv, y_pred=yhat_inv)

        step_error.append([rmse, relative_rmse, smape, mape])

    if doprint:
        if test_cols is not None:
            print("RMSE error for: " + ''.join('{}: {:.2f} || '.format(test_cols[i], step_error[i][0])
                                               for i in range(output_step)))
            print("relative RMSE error for: " + ''.join('{}: {:.2f} || '.format(test_cols[i], step_error[i][1])
                                                        for i in range(output_step)))
            print("sMAPE error for: " + ''.join('{}: {:.2f} || '.format(test_cols[i], step_error[i][2])
                                                for i in range(output_step)))
            print("MAPE for: " + ''.join('{}: {:.2f} || '.format(test_cols[i], step_error[i][3])
                                              for i in range(output_step)))
        else:
            print("RMSE error for: " + ''.join('{}step: {:.2f} || '.format(i + 1, step_error[i][0])
                                               for i in range(output_step)))
            print("relative RMSE error for: " + ''.join('{}step: {:.2f} || '.format(i + 1, step_error[i][1])
                                                        for i in range(output_step)))
            print("sMAPE error for: " + ''.join('{}step: {:.2f} || '.format(i + 1, step_error[i][2])
                                                for i in range(output_step)))
            print("MAPE for: " + ''.join('{}step: {:.2f} || '.format(i + 1, step_error[i][3])
                                              for i in range(output_step)))

    step_error = np.asarray(step_error)
    result = {"RMSE": step_error[:, 0], "relative_RMSE": step_error[:, 1],
              "sMAPE": step_error[:, 2], "MAPE": step_error[:, 3]}
    return result


def get_plot(y_true, y_pred, save_path=None,
             title="Test Predictions vs. Actual",
             y_true_label='actual', y_pred_label='predicted', y_label='predicted value'):
    """
    this function plots prediction values vs real values
    """
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_title(title)
    ax.plot(y_pred, color='red', label=y_pred_label)
    ax.plot(y_true, color='green', label=y_true_label)
    # ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    plt.legend(loc='lower right')
    plt.xlabel('day')
    plt.ylabel(y_label)
    plt.show()
    if save_path:
        plt.savefig(save_path)
    return fig


def get_subplot(y_true, y_pred, subplots, save_path=None, title="Test Predictions vs. Actual"):
    """
    this function plots prediction values vs real values
    """
    fig, ax = plt.subplots(subplots, figsize=(15, 4))
    fig.suptitle(title)

    sub_len = round(len(y_true)/subplots)
    prev = 0
    for i in range(subplots):
        start = prev
        end = prev + sub_len
        prev = end
        ax[i].plot(y_pred[start:end], color='red', marker='o', label='predicted')
        ax[i].plot(y_true[start:end], color='green', marker='o', label='actual')
        # ax[i].set_xlim(xmin=0)
        ax[i].set_ylim(ymin=0)

    plt.legend(loc="upper right")
    plt.xlabel('day')
    plt.ylabel('predicted value')
    plt.show()
    if save_path:
        plt.savefig(save_path)
    return fig


def MASE(training_series, testing_series, prediction_series, freq=1):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.

    See "Another look at measures of forecast accuracy", Rob J Hyndman
    datadatadatadatadata
    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.

    """
    # print ("Needs to be tested."
    n = training_series.shape[0]
    d = np.abs(np.diff(training_series, freq)).sum() / (n - freq)

    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d


def sMAPE(y_true, y_pred):
    """
    Computes the sMAPE ERROR
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_index = np.where(y_true != 0)[0]
    y_true = y_true[nonzero_index]
    y_pred = y_pred[nonzero_index]

    diff = y_true - y_pred
    notnan_index = np.where(~np.isnan(diff))

    diff = diff[notnan_index]
    y_true = y_true[notnan_index]
    y_pred = y_pred[notnan_index]

    if len(y_true) == 0:
        return -1

    sMAPE = 200 / len(y_true) * np.sum(np.divide(np.absolute(diff),
                                                 np.absolute(y_true) + np.absolute(y_pred)))
    return sMAPE


def MAPE(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_index = np.where(y_true != 0)[0]
    y_true = y_true[nonzero_index]
    y_pred = y_pred[nonzero_index]

    diff = y_true - y_pred
    notnan_index = np.where(~np.isnan(diff))

    diff = diff[notnan_index]
    y_true = y_true[notnan_index]

    if len(y_true) == 0:
        return -1

    mape_res = np.mean(np.abs(diff / y_true)) * 100

    return mape_res


def MSE(y_true, y_pred):
    """
    MSE Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_index = np.where(y_true != 0)[0]
    y_true = y_true[nonzero_index]
    y_pred = y_pred[nonzero_index]

    diff = y_true - y_pred
    notnan_index = np.where(~np.isnan(diff))

    diff = diff[notnan_index]

    if len(diff) == 0:
        return -1

    mse = np.mean(diff ** 2)
    return mse


def RMSE(y_true, y_pred):
    """
    RMSE Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_index = np.where(y_true != 0)[0]
    y_true = y_true[nonzero_index]
    y_pred = y_pred[nonzero_index]

    diff = y_true - y_pred
    notnan_index = np.where(~np.isnan(diff))

    diff = diff[notnan_index]

    if len(diff) == 0:
        return -1

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    return rmse


# Relative RMSE
def RRMSE(y_true, y_pred):
    """
    RRMSE Error
    """
    relative_mse = np.mean(np.divide((y_pred - y_true) ** 2, y_true ** 2))
    relative_rmse = np.sqrt(relative_mse)
    return relative_rmse


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_index = np.where(y_true != 0)[0]
    y_true = y_true[nonzero_index]
    y_pred = y_pred[nonzero_index]

    diff = y_true - y_pred
    notnan_index = np.where(~np.isnan(diff))

    diff = diff[notnan_index]
    y_true = y_true[notnan_index]

    if len(y_true) == 0:
        return -1

    mape_res = np.mean(np.abs(diff))

    return mape_res


def pearson_correlation(y_true, y_pred):
    """
    pearson_correlation of two vectors
    """
    r, p = stats.pearsonr(y_true, y_pred)
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")
    return r

