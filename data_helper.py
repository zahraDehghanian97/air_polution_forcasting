import pickle
import os
import numpy as np
from pandas import concat
import pandas as pd
import converter

from db_helper import DbHelper


def to_float(mystr):
    """
    cast the input from string to float number

    input
    -----
    a floating point number in string formant

    return
    ______
    a floating point number
    """
    if mystr == "None" or len(mystr) == 0:
        return np.nan
    else:
        try:
            res = float(mystr)
        except:
            res = np.nan
        return res


def to_int(mystr):
    """
    cast the input from string to integer number

    input
    -----
    an integer number in string formant

    return
    ______
    an integer number
    """
    if mystr == "None":
        return np.nan
    else:
        return int(mystr)


def convert_to_jalili_date(x):
    """
    convert a date input from gregorian to jalili format

    input
    -----
    a datetime formant input in gregorian formant

    return
    ------
    a string date in jalali format
    """
    year = x.year
    month = x.month
    day = x.day
    g = converter.gregorian_to_jalali(gy=year, gm=month, gd=day)
    for i, val in enumerate(g):
        val = str(val)
        if len(val) < 2:
            val = "0" + val
        g[i] = val
    jalili = '-'.join(i for i in g)
    return jalili


def missing_value_imputation(data_df):
    """
    filling missing values in a dataframe using linear, ffill and bfill strategies.
    """
    # if dropna is True we will drop nan values in target column
    data_df = data_df.interpolate(method='linear', limit_direction='both')
    data_df = data_df.fillna(method="ffill")
    data_df = data_df.fillna(method="bfill")
    data_df = data_df.fillna(0)
    return data_df


def export_db_pollution_aqi_data(from_date):
    """
    export AQI data from DataBase from a specific date

    input:
    -----
    from_date: the date that we want to extract information from

    return:
    ------
    AQI data in a DataFrame format
    """
    mydb = DbHelper()
    mydb.connect2db()

    # export pollution data from DB
    query = """ SELECT [Date], [StationId], [CO], [O3_1], [O3_8], [O3], [NO2], [SO2], [PM10], [PM2_5], [AQI] 
                FROM [AirControlCenter].[dbo].[AQIData]
                WHERE [Date] > ?
            """
    values = (from_date,)
    data_rows = pd.read_sql_query(query, mydb.conn, params=values)

    mydb.close()
    return data_rows


def export_db_weather_data(from_date):
    """
    export weather information data from DataBase from a specific date

    input:
    ------
    from_date: the date that we want to extract information from

    return:
    -------
    weather information data in a DataFrame format
    """
    mydb = DbHelper()
    mydb.connect2db()

    query = """ SELECT [Id], [StationName], [Status], [Date], [WindSpeed2], [WindDegree2], [WindDirection2],
                       [WindSpeed10], [WindDegree10], [WindDirection10], [Temperture], [RelativeHumidity],
                       [GlobalRadiation], [DewpointTemperature], [Pressure], [Visibility], [Cloud]
                FROM [AirControlCenter].[dbo].[StationWeatherData] 
                WHERE [Date] > ?              
            """
    values = (from_date,)

    mydb.cursor.execute(query, values)
    data_rows = mydb.cursor.fetchall()

    mydb.close()
    return data_rows


def prepare_pollution_aqi_data(from_date):
    """
    extract AQI data from DataBase and do some preprocessing step on it

    input:
    ------
    from_date: the date that we want to extract information from

    return:
    ------
    AQI data in a DataFrame format
    """
    pollution_df = export_db_pollution_aqi_data(from_date)
    # save_var("row_data.pckl", data_rows)
    # data_rows = load_var("row_data.pckl")

    pollution_df['date'] = pd.to_datetime(pollution_df['Date'])
    pollution_df.sort_values(['date'], inplace=True)
    pollution_df = pollution_df.set_index('date')
    pollution_df = pollution_df.drop(['Date'], axis=1)

    pollution_df['StationId'].fillna(1000, inplace=True)

    station_id_list = set(pollution_df['StationId'].values)

    station_data = []
    tags = []
    column = pollution_df.columns.values
    for i, id in enumerate(station_id_list):
        print("step is {} and id is {}".format(i, id))
        tmp_df = pollution_df.loc[pollution_df['StationId'] == id]

        match_timestamp = "11:00:00"
        tmp_df = tmp_df.loc[tmp_df.index.strftime("%H:%M:%S") == match_timestamp]

        tmp_df = tmp_df.resample('D').mean()
        tmp_df = tmp_df.drop(['StationId'], axis=1)
        c = tmp_df.columns
        d = [c[i] + "_S" + str(int(id)) for i in range(len(c))]
        print(d)
        station_data.append(tmp_df)
        tags.extend(d)

    final = pd.concat(station_data, axis=1, sort=False)
    final.columns = tags

    columns_to_retain = []
    station_dict = load_station_list()
    valid_stations = ["S{}".format(key) for key in station_dict]

    for t in tags:
        station_num = t.split("_")[-1]
        if station_num in valid_stations:
            columns_to_retain.append(t)

    final = final[columns_to_retain]

    # final.to_excel("final_data.xlsx")
    # save_var("final_data.pckl", final)
    return final


def prepare_weather_data(from_date):
    """
    export weather information data from DataBase from a specific date and do some preprocessing step on it

    input:
    ------
    from_date: the date that we want to extract information from

    return:
    -------
    weather information data in a DataFrame format
    """
    weather_rows = export_db_weather_data(from_date)
    # save_var("weather_data.pckl", weather_rows)
    # weather_rows = load_var("weather_data.pckl")

    column_tags = ["Date", "WindSpeed10", "WindDegree10", "Temperture", "RelativeHumidity",
                   "DewpointTemperature", "Pressure", "Visibility"]

    data_tuples = []
    for i, el in enumerate(weather_rows):
        (Id, StationName, Status, Date, WindSpeed2, WindDegree2, WindDirection2, WindSpeed10,
         WindDegree10, WindDirection10, Temperture, RelativeHumidity, GlobalRadiation,
         DewpointTemperature, Pressure, Visibility, Cloud) = el

        tmp_list = [WindSpeed10, WindDegree10, Temperture, RelativeHumidity, DewpointTemperature, Pressure, Visibility]
        for i, el in enumerate(tmp_list):
            try:
                el = float(el)
            except:
                el = np.nan

            tmp_list[i] = el

        tmp_list = [Date] + tmp_list
        data_tuples.append(tmp_list)
        # if i == 1000:
        #     break
    weather_df = pd.DataFrame(data_tuples, columns=column_tags)
    weather_df['date'] = pd.to_datetime(weather_df['Date'])
    weather_df.sort_values(['date'], inplace=True)
    weather_df = weather_df.set_index('date')
    weather_df = weather_df.drop(['Date'], axis=1)

    weather_df = weather_df.resample('D').mean()

    weather_cols = ["WindSpeed10", "WindDegree10", "Temperture",
                    "RelativeHumidity", "DewpointTemperature"]
    weather_df = weather_df[weather_cols]

    return weather_df


def load_station_list():
    """
    reading station information from a file named "station_id.xlsx"

    returns:
    -------
    returns a dictionary data structure which assigns each station to its ID
    """
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    station_df = pd.read_excel(dir_path + "station_id.xlsx")
    station_dict = {}
    for i, row in station_df.iterrows():
        station_dict[row['id']] = row['name']
    return station_dict
