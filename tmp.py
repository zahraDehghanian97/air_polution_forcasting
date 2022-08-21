import pandas as pd
import tools
import numpy as np

all_results_without = tools.load_var("output/all_results_without_features.pckl")
all_results_with = tools.load_var("output/all_results_with_additional_features.pckl")

all_res = []
for station in all_results_without:
    without_holiday = all_results_without[station]
    with_holiday = all_results_with[station]

    index = []
    res = []

    for M in without_holiday:
        if M == "real_y_1":
            continue

        res_without = tools.get_report(y_true=without_holiday["real_y_1"].values.reshape(-1, 1),
                               y_pred=without_holiday[M].values.reshape(-1, 1),
                               doprint=False)
        rmse_without = res_without['RMSE'][0]
        mape_without = res_without["MAPE"][0]
        res.append(rmse_without)
        index.append("{}".format(M))

        # -------------------------------------------------------------------------------
        res_with = tools.get_report(y_true=with_holiday["real_y_1"].values.reshape(-1, 1),
                               y_pred=with_holiday[M].values.reshape(-1, 1),
                               doprint=False)
        rmse_with = res_with['RMSE'][0]
        mape_with = res_with["MAPE"][0]
        res.append(rmse_with)
        index.append("{}_{}".format(M, "Holiday"))

    tmp_df = pd.DataFrame(res, index= index, columns=[[station]])
    all_res.append(tmp_df)

all_res = pd.concat(all_res, axis=1)
all_res.to_excel("output/rmse_holiday.xlsx")





# all_results = all_results["AQI_S22_1"]
#
# res = tools.get_report(y_true=all_results["real_y_1"].values.reshape(-1, 1),
#                        y_pred=all_results["hybrid_svr_1"].values.reshape(-1, 1),
#                        doprint=False)
# rmse = res['RMSE']
# mape = res["MAPE"]
# print("hybrid_svr rmse is: {}, and mape is: {}".format(rmse, mape))
#
# MLR = pd.read_excel("output/MLR-AQI_S22-MC-False.xlsx")
# MLR['date'] = pd.to_datetime(MLR['date'])
# MLR.sort_values(['date'], inplace=True)
# MLR = MLR.set_index('date')
# res = tools.get_report(y_true=MLR["real_y"].values.reshape(-1, 1),
#                        y_pred=MLR['predicted_y'].values.reshape(-1, 1),
#                        doprint=False)
# rmse = res['RMSE']
# mape = res["MAPE"]
# print("MLR rmse is: {}, and mape is: {}".format(rmse, mape))
#
# suse = 5
