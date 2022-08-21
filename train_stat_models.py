from statistical_models import ARIMAX_MODEL, ARIMA_MODEL, EXP_SMOOTHING_MODEL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
import tools as tl
import os


def model_fit(X_train, X_test, params, is_save):
    """
    this function fits an arima or ES model on train data

    inputs:
    X_train, X_test: a DataFrame of pollution data
    params: configuration parameters. an object of the model_parameters class
    is_save: a boolean flag to save or not to save the trained model
    """
    # build model
    exog = None
    if params.model_name.startswith("ARIMAX"):
        model = ARIMAX_MODEL(history=X_train['y'].tolist(), exog=X_train[params.exog_lst[0]].tolist(),
                             exog_forcasted=X_test[params.exog_lst[0]].tolist(), parameter_order=(2, 1, 0))
        model, predictions, test = model.test_vs_predicted(X_test['y'].tolist(), step=params.output_time_steps)
        exog = pd.concat([X_train[params.exog_lst[0]], X_test[params.exog_lst[0]]])

    elif params.model_name.startswith("ARIMA"):
        model = ARIMA_MODEL(history=X_train['y'].tolist(), parameter_order=(2, 1, 0))
        model, predictions, test = model.test_vs_predicted(X_test['y'].tolist(), step=params.output_time_steps)
    else:
        model = EXP_SMOOTHING_MODEL(X_train['y'].tolist())
        model, predictions, test = model.test_vs_predicted(X_test['y'].tolist(), step=params.output_time_steps)

    if params.save_model_path and is_save:
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
        history = pd.concat([X_train['y'], X_test['y']])
        tl.save_var("{}.pckl".format(dir_path + params.save_model_path), model)
        tl.save_var("{}_meta_data.pckl".format(dir_path + params.save_model_path), [params, history, exog])

    return model, predictions, test


def train_arima_es(df, weather_df, mask_df, params):
    """
    this function builds a prediction model based on arima or ES.

    input:
    ------
    df: a dataframe containing the pollution information of all stations
    weather_df: a dataframe containing the weather information
    mask_df: a dataframe which show the location of missing values in df
    params: model configuration parameters. an object of model_parameters class
    """
    notnull_list = mask_df[params.target].values.squeeze()

    if params.model_name == "ARIMA" or params.model_name == "ES":
        df_target = df[params.target]
        df_target.columns = ['y']
        # df_target.set_index('ds', inplace=True)
        df_target.dropna(inplace=True)
        # df_target.reset_index(inplace=True)
        # del df_target['date']
    else:
        df = df.merge(weather_df, how='left', left_index=True, right_index=True)
        # df = pd.merge([df, weather_df], axis=1)

        df_target = df[params.target + params.exog_lst]
        df_target.columns = ['y'] + params.exog_lst
        # df_target.set_index('ds', inplace=True)
        df_target.dropna(inplace=True)
        # df_target.reset_index(inplace=True)
        # del df_target['date']

    cut = int(df_target.shape[0] - params.cut)

    # Train test split
    train = df_target[0:cut]
    test = df_target[cut:]
    notnull_list_test = np.array(notnull_list[cut:], dtype=bool)

    # log-transform
    train['y'] = np.log1p(train.y)
    test['y'] = np.log1p(test.y)

    if params.exog_lst is not None:
        train[params.exog_lst] = np.log1p(train[params.exog_lst])
        test[params.exog_lst] = np.log1p(test[params.exog_lst])

    date_index_test = test.index.tolist()
    model, preds, tsts = model_fit(X_train=train, X_test=test, params=params, is_save=True)
    date_index_test = list(compress(date_index_test, notnull_list_test))
    preds = list(compress(preds, notnull_list_test))
    tsts = list(compress(tsts, notnull_list_test))

    inv_y_test = np.asarray(tsts)
    inv_yhat_test = np.asarray(preds)

    y_y_hat = np.concatenate([inv_y_test, inv_yhat_test], axis=1)

    real_col = ["real_y_{}".format(i) for i in range(1, params.output_time_steps+1)]
    pred_col = ["{}_y_{}".format(params.model_name, i) for i in range(1, params.output_time_steps+1)]

    cols = real_col + pred_col

    y_y_hat = pd.DataFrame(y_y_hat, columns=cols, index=date_index_test)
    y_y_hat = np.expm1(y_y_hat)

    return y_y_hat
