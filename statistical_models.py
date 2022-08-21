import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings
warnings.filterwarnings('ignore')
import os


class ARIMAX_MODEL:
    def __init__(self, history=[], exog=None, exog_forcasted=None, parameter_order=(0, 0, 0)):
        """
        Initial parameter settings for ARIMAX class
        @param self:
        @param history: list of previous observed samples
        @param exog: exogenous variable in ARIMAX model
        @param exog_forcasted: forcasted samples for exog variables
        @param parameter_order: ARIMAX parameters
        """
        self.history = history
        self.exog = exog
        self.exog_forcasted = exog_forcasted
        self.parameter = parameter_order
        self.__model = None

    def fit(self):
        """
        fitting the model on training data set
        """
        model = ARIMA(endog=self.history, exog=self.exog, order=self.parameter)
        model_fit = model.fit(disp=0)
        self.__model = model_fit
        return self

    def predict(self, step=1):
        """
        prediction for number of steps
        @param self:
        @param step: forward prediction steps
        @return: list of predicted values
        """
        output = self.__model.forecast(steps=step, exog=self.exog_forcasted[0:step])[0]
        yhats = output.tolist()
        return yhats

    def test_vs_predicted(self, test, step=1):
        """
        computes the list of prediction for the list of test data
        @param self:
        @param test: list of test values
        @param step: forward prediction steps
        @return: (model,predictions, test) pairs
        """

        history_stack = self.history
        exog_vars = self.exog
        exog_var_forcasted = self.exog_forcasted
        predictions = list()
        real_vals = list()

        lst_stp = list(range(len(test)))
        for i in range(step-1):
            test.append(0)
            exog_var_forcasted.append(exog_var_forcasted[-1])

        for t in lst_stp:
            model = ARIMA(endog=history_stack, exog=exog_vars, order=self.parameter)  # order=(2,1,1)
            model = model.fit(disp=0)
            output = model.forecast(steps=step, exog=exog_var_forcasted[t:t+step])[0]
            yhat = output.tolist()
            predictions.append(yhat)
            real_vals.append(test[t:t + step])

            obs = test[t]
            history_stack.append(obs)
            exog_vars.append(self.exog_forcasted[t])
        model = ARIMA(endog=history_stack, exog=exog_vars, order=self.parameter)  # order=(2,1,1)
        model = model.fit(disp=0)
        return model, predictions, real_vals


class ARIMA_MODEL:
    def __init__(self, history=[], parameter_order=(0, 0, 0)):
        """
        Initial parameter settings for ARIMA class
        @param self:
        @param history: list of previous observed samples
        @param parameter_order: ARIMA parameters
        """
        self.history = history
        self.parameter = parameter_order
        self.__model = None

    def fit(self):
        """
        fitting the model on training data set
        """
        model = ARIMA(self.history, order=self.parameter)
        model_fit = model.fit(disp=0)
        self.__model = model_fit
        return self

    def predict(self, step=1):
        """
        prediction for number of steps
        @param self:
        @param step: forward prediction steps
        @return: list of predicted values
        """

        output = self.__model.forecast(steps=step)[0]
        yhats = output.tolist()
        return yhats

    def test_vs_predicted(self, test, step=1):
        """
        computes the list of prediction for the list of test data
        @param self:
        @param test: list of test values
        @param step: forward prediction steps
        @return: (model,predictions, test) pairs
        """

        history_stack = self.history
        predictions = list()
        real_vals = list()

        lst_stp = list(range(len(test)))
        for i in range(step-1):
            test.append(0)

        for t in lst_stp:
            model = ARIMA(history_stack, self.parameter)  # order=(2,1,1)
            model = model.fit(disp=0)

            # if t % step == 0:
            output = model.forecast(steps=step)[0]
            yhat = output.tolist()
            predictions.append(yhat)
            real_vals.append(test[t:t + step])

            obs = test[t]
            history_stack.append(obs)
            # print('predicted=,', yhat, 'expected= ', obs)

        model = ARIMA(history_stack, self.parameter)  # order=(2,1,1)
        model = model.fit(disp=0)

        # if len(predictions) <= len(test):
        #     predictions = predictions + ((len(test) - len(predictions)) * [0])
        # else:
        #     predictions = predictions[:len(test)]
        return model, predictions, real_vals


class EXP_SMOOTHING_MODEL:
    def __init__(self, history=[], smth_pr=None):
        """
        Initial parameter settings for ETS class
        @param self:
        @param history: list of previous observed samples
        @param smth_pr: smoothing parameter in ETS model
        """
        self.history = history
        self.__model = None
        self.smoothing_parameter = smth_pr

    def fit(self):
        """
        fitting the model on training data set
        """
        model = SimpleExpSmoothing(self.history)
        model_fit = model.fit(smoothing_level = self.smoothing_parameter)
        self.__model = model_fit
        return self

    def predict(self, step=1):
        """
        prediction for number of steps
        @param self:
        @param step: forward prediction steps
        @return: list of predicted values
        """

        output = self.__model.forecast()[0]
        yhats = output.tolist()
        return yhats

    def test_vs_predicted(self, test, step=1):
        """
        computes the list of prediction for the list of test data
        @param self:
        @param test: list of test values
        @param step: forward prediction steps
        @return: (model,predictions, test) pairs
        """

        history_stack = self.history
        predictions = list()
        real_vals = list()
        # lst_stp = [step * i for i in range(int(len(test) / step))]
        lst_stp = list(range(len(test)))
        for i in range(step-1):
            test.append(0)

        for t in lst_stp:
            step_pred = []
            for inc in range(step):
                model = SimpleExpSmoothing(history_stack)
                model = model.fit(smoothing_level=self.smoothing_parameter)
                yhat = model.forecast()[0]
                history_stack.append(yhat)
                step_pred.append(yhat)
            predictions.append(step_pred)
            obs_lst = test[t:t + step]
            real_vals.append(obs_lst)
            history_stack = history_stack[:-step] + [test[t]]

        model = SimpleExpSmoothing(history_stack)
        model = model.fit(smoothing_level=self.smoothing_parameter)

        return model, predictions, real_vals
