import numpy as np
import pandas as pd
import copy
import tools
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os


class MVIParameters:
    """
    configuration parameters for missing value imputation models
    """
    def __init__(self):
        self.target = None
        self.feature_count = 10

        # missing value imputation strategy (linear, None)
        self.mvi_strategy = "linear"
        self.dir_path = os.path.dirname(os.path.realpath(__file__)) + "/imputer/"


def missing_value_imputation(X_missing, method, impute_estimator=None):
    """
    this function fills missing values

    input:
    ------
    X_missing: the input matrix to fill missing values
    method: missing value imputation method
    impute_estimator: if the imputation is based on regression we should give the estimator of the model
    """
    imp = None
    if method in ["BayesianRidge", "DecisionTreeRegressor",
                  "ExtraTreesRegressor", "KNeighborsRegressor"]:
        imp = IterativeImputer(max_iter=20, random_state=0, estimator=impute_estimator)
        imp.fit(X_missing.values)

        predicted = imp.transform(X_missing.values)
    elif method in ["linear", "cubic"]:
        predicted = X_missing.interpolate(method=method).values
    elif method in ["ffill", "bfill"]:
        predicted = X_missing.fillna(method=method).values
    else:
        predicted = X_missing.fillna(X_missing.rolling(5, min_periods=1, center=True).mean()).values

    return predicted, imp


def fill_recursivly(data_df, best_method_list):
    """
    this function fills the missing values of a dataframe with best methods
    """
    best_mape = ""
    best_method = ""
    imputer_list = []
    for method in best_method_list:
        best_mape = best_mape + "{:0.2f}__".format(method[0])
        best_method = best_method + "{}__".format(method[1])
        predicted, imputer = missing_value_imputation(data_df.copy(), method[1], method[2])
        imputer_list.append((method[1], imputer))

        data_df.iloc[:, :] = predicted
        if data_df.isnull().values.any():
            continue
        else:
            return predicted, best_mape, best_method, imputer_list


def fill_missing_values(df, params):
    """

    """
    all_results = {}
    all_methods = []

    n_samples, n_features = df.shape

    tags = df.columns
    target_index = np.where(tags.values == params.target)
    target_index = target_index[0][0]
    tmp_target = copy.deepcopy(df[params.target])

    not_null_index = tmp_target.notna().values.squeeze()

    rng = np.random.RandomState(0)
    # Add a single missing value to each row
    missing_samples = np.arange(n_samples)
    missing_samples = missing_samples[not_null_index]
    n_samples = round(len(missing_samples) * 0.15)
    missing_indexes = rng.choice(missing_samples, n_samples, replace=True)

    X_missing = df.copy()
    X_missing.iloc[missing_indexes, target_index] = np.nan
    true_values = df.values[missing_indexes, target_index]

    if len(true_values) == 0:
        predicted, imputer = missing_value_imputation(df.copy(), "BayesianRidge", BayesianRidge())
        predicted = predicted[:, target_index]
        best_mape = "-1"
        best_method = "not tested"
        return predicted, best_mape, best_method, None

    estimators = [BayesianRidge(),
                  DecisionTreeRegressor(max_features='sqrt', random_state=0),
                  ExtraTreesRegressor(n_estimators=10, random_state=0),
                  KNeighborsRegressor(n_neighbors=15)]

    best_method = []

    for impute_estimator in estimators:
        method = impute_estimator.__class__.__name__
        try:
            predicted, imputer = missing_value_imputation(X_missing.copy(), method, impute_estimator)
            predicted_values = predicted[missing_indexes, target_index]
        except:
            print("exception accured")
            continue

        mse = tools.MSE(y_true=true_values, y_pred=predicted_values)
        rmse = tools.RMSE(y_true=true_values, y_pred=predicted_values)
        mape = tools.MAPE(y_true=true_values, y_pred=predicted_values)
        smape = tools.sMAPE(y_true=true_values, y_pred=predicted_values)

        all_results[impute_estimator.__class__.__name__] = [mse, rmse, mape, smape]
        all_methods.append(impute_estimator.__class__.__name__)

        best_method.append((mape, method, impute_estimator))

    all_methods = ["linear", "cubic", "ffill", "bfill", "rolling5"] + all_methods
    for method in ["linear", "cubic", "ffill", "bfill", "rolling5"]:
        try:
            predicted, imputer = missing_value_imputation(X_missing.copy(), method)
            predicted_values = predicted[missing_indexes, target_index]
        except:
            print("exception accured for {}".format(method))
            continue
        mse = tools.MSE(y_true=true_values, y_pred=predicted_values)
        rmse = tools.RMSE(y_true=true_values, y_pred=predicted_values)
        mape = tools.MAPE(y_true=true_values, y_pred=predicted_values)
        smape = tools.sMAPE(y_true=true_values, y_pred=predicted_values)

        all_results[method] = [mse, rmse, mape, smape]

        best_method.append((mape, method, None))
    best_method = sorted(best_method, key=lambda x: x[0])

    predicted, best_mape, best_method1, imputer_list = fill_recursivly(df.copy(), best_method)
    predicted = predicted[:, target_index]

    return predicted, best_mape, best_method1, imputer_list


def fill_data(df, params):
    tags = df.columns.tolist()

    corrMatrix = df.corr().abs()

    final_df = df.copy()
    acc_list = []

    for tmp_target in tags:
        # tmp_target = "PM10_S20"    #"PM_2.5_S2"
        params.target = [tmp_target]
        target = params.target[0]

        cors = []
        for t in tags:
            tmp_corr = corrMatrix[target][t]
            cors.append((tmp_corr, t))

        corr_list = sorted(cors, key=lambda x: x[0], reverse=True)
        corr_list = corr_list[:params.feature_count]
        correlated_tags = [t[1] for t in corr_list]
        if target not in correlated_tags:
            correlated_tags = [target] + correlated_tags
            correlated_tags = correlated_tags[:-1]

        tags_to_remove = []
        for t in correlated_tags:
            tmp_list = df[t].copy().values
            if np.isnan(tmp_list).all():
                tags_to_remove.append(t)
                print("all nan")

        correlated_tags = list(set(correlated_tags) - set(tags_to_remove))
        if (params.target[0] in tags_to_remove) or (len(correlated_tags) == 0):
            acc_list.append((target, "Cant fill", "Nothing"))
            # print("kar kharab ast")
            continue

        selected_df = df[correlated_tags].copy()

        predicted_values, best_mape, best_method, imputer_list = fill_missing_values(selected_df, params)
        if imputer_list is not None:
            tools.save_var("{}{}.pckl".format(params.dir_path, tmp_target), [imputer_list, correlated_tags])
        print(target, best_method, best_mape)

        null_idx = final_df[target].isnull().values.squeeze()
        final_df[target].loc[null_idx] = predicted_values[null_idx]
        acc_list.append((target, best_mape, best_method))
        # break

    summary_df = pd.DataFrame(acc_list, columns=['column', 'MAPE', 'Method'])
    return final_df, summary_df


def train_mvi_models(pollution_df, weather_df):
    mvi_params = MVIParameters()

    # make a mask df of data
    mask_df = pd.notnull(pollution_df)
    mask_df = mask_df.astype(int)

    # fill weather_df missing values
    tags = weather_df.columns
    acc_list = []
    correlated_tags = tags
    for i, tmp_target in enumerate(tags):
        mvi_params.target = [tmp_target]
        target = mvi_params.target[0]

        predicted_values, best_mape, best_method, imputer_list = fill_missing_values(weather_df, mvi_params)
        tools.save_var("{}{}.pckl".format(mvi_params.dir_path, tmp_target), [imputer_list, correlated_tags])
        weather_df[target] = predicted_values
        print(target, best_method, best_mape)
        acc_list.append((target, best_mape, best_method))

    # make a mask df of weather data
    bool_weather_df = pd.notnull(weather_df)
    bool_weather_df = bool_weather_df.astype(int)

    # fill df missing values
    pollution_df, summary_df = fill_data(pollution_df, mvi_params)


    # concat df with weather df
    res_df = pd.concat([pollution_df, weather_df], axis=1, join="outer")
    res_bool_df = pd.concat([mask_df, bool_weather_df], axis=1, join="outer")

    tags = res_df.columns
    res_bool_df = res_bool_df[tags]

    summary_df.to_excel("output/final_summary_res_AQIData.xlsx")
    res_df.to_excel("output/AQIData_weather_filled.xlsx")
    res_bool_df.to_excel("output/AQIData_weather_mask.xlsx")

    return pollution_df, weather_df, mask_df
