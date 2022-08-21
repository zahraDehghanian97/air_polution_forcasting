from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd

import tools


def svr_model(data_df, feature_label, target_label, save_model_path):
    """
    this function builds a prediction model based on SVR.

    input:
    ------
    data_df: a dataframe containing the prediction results of other prediction models
    feature_label: a list containing the selected features for models to predict
    target_label: a string represents the target variable to predict
    save_model_path: a path to save the final trained model
    """
    k_fold = KFold(n_splits=5)
    real_y = []
    predicted_y = []
    date_index = []
    for train_indices, test_indices in k_fold.split(data_df):
        x_data = data_df[feature_label]
        y_data = data_df[target_label]

        X_train = x_data.iloc[train_indices]
        y_train = y_data.iloc[train_indices]

        X_test = x_data.iloc[test_indices]
        Y_test = y_data.iloc[test_indices]
        test_date_index = X_test.index.tolist()

        Svr = SVR(kernel='linear', C=1)
        Svr.fit(X_train.values, y_train.values)

        y_hat = np.reshape(Svr.predict(X_test.values), (-1, 1))
        y_test = np.reshape(Y_test.values, (-1, 1))

        predicted_y.append(y_hat)
        real_y.append(y_test)
        date_index.append(test_date_index)

    y_test = np.concatenate(real_y, axis=0)
    y_hat = np.concatenate(predicted_y, axis=0)
    date_index = np.concatenate(date_index, axis=0)
    df_res = pd.DataFrame(np.concatenate([y_hat, y_test], axis=1),
                          columns=["predicted_value", "real_value"], index=date_index)

    res = tools.get_report(y_true=y_test, y_pred=y_hat, doprint=False)
    rmse = res['RMSE']
    mape = res["MAPE"]

    final_svr = SVR(kernel='linear', C=1)
    X_train = data_df[feature_label]
    y_train = data_df[target_label]
    final_svr.fit(X_train.values, y_train.values)
    tools.save_var("{}.pckl".format(save_model_path), final_svr)
    tools.save_var("{}_meta_data.pckl".format(save_model_path), [feature_label])
    tools.save_var("{}_test_data.pckl".format(save_model_path), [X_test, Y_test])

    return rmse, mape, df_res