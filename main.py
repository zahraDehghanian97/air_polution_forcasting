from db_helper import DbHelper
from tools import save_var, load_var
import pandas as pd
import numpy as np
from forecast_models import ForecastModels
from forecast_models import hybrid_forcast
from tools import get_report
from datetime import timedelta, date as d
import os
import json

import data_helper
import tools


def write2db(db, date_list, stationID, station_name, predicted_values, real_values):
    """
    write AQI prediction values for different stations on DB
    input
    ------
    db: database connection
    date_list: a list containing a date which represents the first day of prediction
    sationID: Represents station as a number
    station_name: The name of the station
    predicted_values: A list containing the prediction values for the target station. it's size is
                      equal to output_time_steps
    real_values: A list containing one value which is the observed real value of yesterday
    """
    for i in range(len(date_list)):
        date = date_list[i]
        jalili = data_helper.convert_to_jalili_date(date)

        date = str(date).split()[0]
        persianDate = jalili

        real_AQI = real_values[i]
        if np.isnan(real_AQI):
            real_AQI = 'Null'

        predict_AQI = predicted_values[i]
        if np.isnan(predict_AQI):
            predict_AQI = 0

        year = int(persianDate.split("-")[0])
        month = int(persianDate.split("-")[1])
        day = int(persianDate.split("-")[2])

        # query = """ INSERT INTO dbo.AQI_real_predict values (?, ?, ?, ?, ?, ?, ?, ?, ?)
        #         """
        # values = (date, persianDate, str(real_AQI), str(predict_AQI), str(stationID), str(station_name), str(day), str(month), str(year))

        query = "INSERT INTO dbo.AQI_real_predict values ('" + date + "','" + persianDate + '\',' + str(real_AQI)\
                + ',' + str(predict_AQI) + ',' + \
                str(stationID) + ",N'" + str(station_name) + "'," + str(day) + "," + str(month) + "," + str(year) + ")"
        print(query)

        db.cursor2.execute(query)
        db.conn_air2.commit()


if __name__ == "__main__":
    # loading configuration parameters from conf.json file
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    json_data = open(dir_path + 'conf.json').read()
    myconfig = json.loads(json_data)
    # ---------------------------------------------------------
    # setting start_time and end_time. Models will use these values to predict future.
    # Interval length should be larger than "input_time_steps"
    start_time = '2021-04-01'
    end_time = str(d.today())
    # end_time = '2021-04-05'
    end_time = str(d.today() + timedelta(days=-1))

    # --------------------------------------------------------------------------------------------------------
    # load and prepare data from database
    pollution_df = data_helper.prepare_pollution_aqi_data(from_date=start_time)
    weather_df = data_helper.prepare_weather_data(from_date=start_time)

    # select and save rows in a specific range of dates
    pollution_df = pollution_df.loc[start_time: end_time]
    weather_df = weather_df.loc[start_time: end_time]

    # if missing value imputation models are already trained use fill_data_frame function to fill missing values
    weather_df = tools.fill_data_frame(weather_df)
    pollution_df = tools.fill_data_frame(pollution_df)

    tools.save_var("tmp_data.pckl", [pollution_df, weather_df])
    # pollution_df, weather_df = tools.load_var("tmp_data.pckl")

    # ------------------------------------------------------------------------------------------------------------------

    # pollution_df, weather_df, mask_df = tools.load_var("data_table_filled.pckl")
    # pollution_df = pollution_df.loc[start_time: end_time]
    # weather_df = weather_df.loc[start_time: end_time]

    # # concat df with weather df
    # res_df = pd.concat([pollution_df, weather_df], axis=1, join="outer")
    # res_df.to_excel("AQIData.xlsx")

    # ------------------------------------------------------------------------------------------------------------------
    # adding features about holidays and weekends to the data
    if myconfig["add_holiday_features"]:
        weather_df = tools.add_holiday_features(weather_df)

    # loading informations of states that we want to predict AQI for.
    station_dict = data_helper.load_station_list()
    station_list = []
    for key in station_dict:
        station_list.append("AQI_S{}".format(key))
    # station_list = ["AQI_S1000"]

    # a list containing the name of the pre-trained models
    model_list = ["BaseLSTM", "LSTMAttention", "fireTSaR", "fireTSaX", "fireTSnR", "fireTSnX", "ARIMAXW", "ARIMAXH", "ARIMA", "ES"]
    # model_list = ["ES"]

    # creating an object from ForecastModels. this class has some methods to load pre-trained models
    # and predict future based on observations
    forcast_models = ForecastModels()

    # frecast future pollution information and write to DataBase
    # creating a connection point to DataBase
    mydb = DbHelper()
    mydb.connect2bd2()

    yhat_all = []
    y_all = []
    # specifies how far we want to predict. this parameter is based on pre-trained models.
    pred_step = myconfig["output_time_steps"]

    # predicting AQI for each station in station list
    for station in station_list:
        # making a list of predicted values and real values
        predicted_y = []
        real_y = []

        # extract station ID from it's name.
        station_id = int(station.split("_")[-1][1:])
        station_name = station_dict[station_id]

        # each model in model list predicts AQI for a specific station. then a hybrid model uses these values to sum up
        for model_name in model_list:
            print("forcasting for model {}".format(model_name))

            # predict AQI for specific station with specific prediction model
            # y_y_hat is a dataframe with one column containing real AQI values and
            # one column containing predicted AQI values
            y_y_hat = forcast_models.predict(station, model_name, pred_step, pollution_df, weather_df)
            if len(y_y_hat) == 0:
                print("there is no such model")
                break

            res_cols = y_y_hat.columns
            real_cols = []
            for c in res_cols:
                if c.startswith("real"):
                    real_cols.append(c)

            real_y.append(y_y_hat[real_cols].copy())
            y_y_hat = y_y_hat.drop(real_cols, axis=1)
            predicted_y.append(y_y_hat)

        if len(real_y) == 0:
            suse = 5
            continue
        predicted_y.append(real_y[0])

        # creating a dataframe with columns representing the predictions of different models
        prediction_df_list = pd.concat(predicted_y, axis=1)
        prediction_df_list.dropna(inplace=True)

        # loading hybrid models (based on svr) to predict future steps.
        df_cols = prediction_df_list.columns
        for step in range(1, pred_step + 1):
            real_label = "real_y_{}".format(step)
            covariate_col = []
            for c in df_cols:
                if c.endswith("_{}".format(step)):
                    covariate_col.append(c)
            y_hat = hybrid_forcast(prediction_df_list[covariate_col].copy(), station, pred_step=step)
            prediction_df_list["hybrid_svr_{}".format(step)] = y_hat

        print("*********************************")
        y_true = prediction_df_list['real_y_1'].values.reshape(-1, 1)
        y_pred = prediction_df_list["hybrid_svr_1"].values.reshape(-1, 1)
        date_list = prediction_df_list.index.tolist()
        get_report(y_true, y_pred)

        # writing prediction values to DB
        # mydb = None
        write2db(db=mydb, date_list=date_list, stationID=station_id, station_name=station_name,
                 predicted_values=np.squeeze(y_pred), real_values=np.squeeze(y_true))

        yhat_all.append(y_pred)
        y_all.append(y_true)

    mydb.closedb2()
    print("the end")