from db_helper import DbHelper
import pandas as pd
import train_mvi_models
from forecast_models import ForecastModels
from data_helper import convert_to_jalili_date
from datetime import timedelta, date as d
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from models import BaseLSTM, BaseAttention
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from itertools import compress
import json

import model_parameters
from train_LSTM_model import train_lstm
from train_firets_model import train_fireTS
from train_stat_models import train_arima_es
from train_svr_model import svr_model
import tools
import data_helper


def train_hybrid_model(prediction_df_list, target, pred_step):
    """
    This function trains the final prediction model (a svr based model for regression)

    inputs:
    prediction_df_list: input data. The collection of predictions with other prediction models in a DataFrame format.
                        columns of the DataFrame represent model names.
                        index of the DataFrame represents the date of the prediction
    target: specifies the target AQI which we want to predict the values
    pred_step: represents the prediction time step
    """
    prediction_df_list.dropna(inplace=True)

    target_label = ["real_y_{}".format(pred_step)]
    feature_label = list(set(prediction_df_list.columns) - set(target_label))
    rmse, mape, df_res = svr_model(prediction_df_list, feature_label,
                                   target_label, save_model_path="models/{}_{}_{}".format("hybrid_svr", target, pred_step))
    prediction_df_list["hybrid_svr_{}".format(pred_step)] = df_res['predicted_value']
    res = tools.get_report(y_true=prediction_df_list[target_label].values.reshape(-1, 1),
                           y_pred=prediction_df_list["hybrid_svr_{}".format(pred_step)].values.reshape(-1, 1),
                           doprint=False)
    return prediction_df_list

    # dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    # hybrid_model_path = dir_path + "models/{}_{}.pckl".format("hybrid_svr", target)
    # hybrid_model_path_meta = dir_path + "models/{}_{}_meta_data.pckl".format("hybrid_svr", target)
    # hybrid_model = tools.load_var(hybrid_model_path)
    # feature_label = tools.load_var(hybrid_model_path_meta)[0]
    #
    # hybrid_model_input = prediction_df_list[feature_label]
    #
    # y_hat = hybrid_model.predict(hybrid_model_input.values)
    # prediction_df_list["hybrid_svr"] = y_hat
    # res = tools.get_report(y_true=prediction_df_list['real_y'].values.reshape(-1, 1),
    #                        y_pred=prediction_df_list["hybrid_svr"].values.reshape(-1, 1),
    #                        doprint=False)


def train_all_models(df, weather_df, mask_df, config):
    """
    This function trains all the prediction models and saves them on hard disk

    input:
    -------
    df: A DataFrame which contains AQI data of different stations on different days
    weather_df: A DataFrame which contains weather information on different days
    mask_df: A DataFrame which specifies the location of missing values in the df
    config: a dictionary containing the configuration parameters

    output:
    -------
    the prediction of each model on test data.
    """
    models_dict = {
                   "BaseLSTM": model_parameters.Params(model_name="BaseLSTM",
                                                       input_time_steps=config["input_time_steps"],
                                                       output_time_steps=config["output_time_steps"],
                                                       future_weather_lookahead=config["future_weather_lookahead"],
                                                       holiday_features=config["add_holiday_features"]),

                   "LSTMAttention": model_parameters.Params(model_name="LSTMAttention",
                                                            input_time_steps=config["input_time_steps"],
                                                            output_time_steps=config["output_time_steps"],
                                                            future_weather_lookahead=config["future_weather_lookahead"],
                                                            holiday_features=config["add_holiday_features"]),

                   "fireTSaR": model_parameters.Params(model_name="fireTSaR", firets_model_name="autoregression",
                                                       firets_base_regressor=RandomForestRegressor(),
                                                       input_time_steps=config["input_time_steps"],
                                                       output_time_steps=config["output_time_steps"],
                                                       future_weather_lookahead=config["future_weather_lookahead"],
                                                       holiday_features=config["add_holiday_features"]),

                   "fireTSaX": model_parameters.Params(model_name="fireTSaX", firets_model_name="autoregression",
                                                       firets_base_regressor=XGBRegressor(),
                                                       input_time_steps=config["input_time_steps"],
                                                       output_time_steps=config["output_time_steps"],
                                                       future_weather_lookahead=config["future_weather_lookahead"],
                                                       holiday_features=config["add_holiday_features"]),

                   "fireTSnR": model_parameters.Params(model_name="fireTSnR",
                                                       firets_model_name="non_linear_autoregression",
                                                       firets_base_regressor=RandomForestRegressor(),
                                                       input_time_steps=config["input_time_steps"],
                                                       output_time_steps=config["output_time_steps"],
                                                       future_weather_lookahead=config["future_weather_lookahead"],
                                                       holiday_features=config["add_holiday_features"]),

                   "fireTSnX": model_parameters.Params(model_name="fireTSnX",
                                                       firets_model_name="non_linear_autoregression",
                                                       firets_base_regressor=XGBRegressor(),
                                                       input_time_steps=config["input_time_steps"],
                                                       output_time_steps=config["output_time_steps"],
                                                       future_weather_lookahead=config["future_weather_lookahead"],
                                                       holiday_features=config["add_holiday_features"]),

                   "ES": model_parameters.StatParams(model_name="ES",
                                                     output_time_steps=config["output_time_steps"],
                                                     holiday_features=config["add_holiday_features"]),

                   "ARIMA": model_parameters.StatParams(model_name="ARIMA",
                                                        output_time_steps=config["output_time_steps"],
                                                        holiday_features=config["add_holiday_features"]),

                   "ARIMAXW": model_parameters.StatParams(model_name="ARIMAXW",
                                                          exog_lst=["WindSpeed10"],
                                                          output_time_steps=config["output_time_steps"],
                                                          holiday_features=config["add_holiday_features"]),

                   "ARIMAXH": model_parameters.StatParams(model_name="ARIMAXH",
                                                          exog_lst=["RelativeHumidity"],
                                                          output_time_steps=config["output_time_steps"],
                                                          holiday_features=config["add_holiday_features"])
    }

    target_stations = data_helper.load_station_list()
    target_list = ["AQI_S{}".format(key) for key in target_stations]

    all_results = {}
    pred_step = 0
    for target in target_list:
        prediction_df_list = []
        real_value_list = []
        for model_name in models_dict:
            # y_y_hat = None
            # if target in ["AQI_S1", "AQI_S5"]:
            #     suse = 5
            #     continue

            params = models_dict[model_name]
            pred_step = params.output_time_steps
            print("training for target {}".format(target))
            params.target = [target]
            params.save_model_path = "models/{}_{}".format(model_name, target)

            y_y_hat = None
            if model_name in ["BaseLSTM", "LSTMAttention"]:
                y_y_hat = train_lstm(df.copy(), weather_df.copy(), mask_df.copy(), params)
            elif model_name in ["fireTSaR", "fireTSaX", "fireTSnR", "fireTSnX"]:
                y_y_hat = train_fireTS(df.copy(), weather_df.copy(), mask_df.copy(), params)
            elif model_name in ["ARIMA", "ARIMAXH", "ARIMAXW", "ES"]:
                y_y_hat = train_arima_es(df.copy(), weather_df.copy(), mask_df.copy(), params)

            if y_y_hat is None:
                suse = 5
                break

            res_cols = y_y_hat.columns
            real_cols = []
            for c in res_cols:
                if c.startswith("real"):
                    real_cols.append(c)

            real_value_list.append(y_y_hat[real_cols].copy())
            y_y_hat = y_y_hat.drop(real_cols, axis=1)

            prediction_df_list.append(y_y_hat)

        if y_y_hat is None:
            suse = 5
            continue

        prediction_df_list.append(real_value_list[0])
        prediction_df_list = pd.concat(prediction_df_list, axis=1)

        df_cols = prediction_df_list.columns

        for pred_step in range(1, pred_step+1):
            real_label = "real_y_{}".format(pred_step)
            covariate_col = []
            for c in df_cols:
                if c.endswith("_{}".format(pred_step)):
                    covariate_col.append(c)
            prediction_res_df = train_hybrid_model(prediction_df_list[covariate_col].copy(), target=target,
                                                   pred_step=pred_step)
            all_results["{}_{}".format(target, pred_step)] = prediction_res_df

    return all_results


if __name__ == "__main__":
    # loading configuration parameters from conf.json file
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    json_data = open(dir_path + 'conf.json').read()
    myconfig = json.loads(json_data)

    # -----------------------------------------------------------------------------------------------------------
    # setting start_time and end_time. Models will use these values to predict future.
    # Interval length should be larger than "input_time_steps"
    start_time = '2014-01-01'
    end_time = str(d.today())
    # end_time = '2021-01-08'

    # load and prepare data from database
    pollution_df = data_helper.prepare_pollution_aqi_data(from_date=start_time)
    weather_df = data_helper.prepare_weather_data(from_date=start_time)

    # select and save rows in a specific range of dates
    pollution_df = pollution_df.loc[start_time: end_time]
    weather_df = weather_df.loc[start_time: end_time]
    # tools.save_var("data_table.pckl", [pollution_df, weather_df])

    # -------------------------------------------------
    # # if missing value imputation models are already trained use fill_data_frame function to fill missing values
    # pollution_df, weather_df = tools.load_var("data_table.pckl")
    # weather_df = tools.fill_data_frame(weather_df)
    # pollution_df = tools.fill_data_frame(pollution_df)
    # -------------------------------------------------

    # train models for missing value imputation
    df, weather_df, mask_df = train_mvi_models.train_mvi_models(pollution_df, weather_df)
    tools.save_var("data_table_filled.pckl", [df, weather_df, mask_df])
    # -----------------------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------------------
    # df, weather_df, mask_df = tools.load_var("data_table_filled.pckl")
    # -----------------------------------------------------------------------------------------------------------
    # adding features about holidays and weekends to the data
    if myconfig["add_holiday_features"]:
        weather_df = tools.add_holiday_features(weather_df)

    all_results = train_all_models(df, weather_df, mask_df, myconfig)
    tools.save_var("all_results.pckl", all_results)

    print("the end")