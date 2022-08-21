# model configuration parameters
class Params:
    def __init__(self, model_name, input_time_steps, output_time_steps, future_weather_lookahead,
                 holiday_features, firets_model_name=None, firets_base_regressor=None):
        self.model_name = model_name
        self.input_time_steps = input_time_steps
        self.output_time_steps = output_time_steps
        self.future_weather_lookahead = future_weather_lookahead

        self.target = ["AQI_S1000"]
        self.feature_count = 5
        self.feature_selection_method = "MC"
        self.holiday_features = holiday_features

        # outlier cleaning
        self.outlier_factor = 1.5
        self.outlier_strategy = None
        self.is_clean_outlier = True
        self.is_exp_smoothing = False

        # network parameters
        self.epochs = 40
        self.batch_size = 50
        self.hidden_dim = 70
        self.input_dropout = 0.0
        self.recurrent_dropout = 0.0
        self.stacket_layer_num = 0
        self.kernel = "LSTM"  # GRU or LSTM or SimpleRNN
        self.loss = 'mae'  # mean_squared_error or mae
        self.optimizer = 'adam'

        # fireTS params
        self.firets_model_name = firets_model_name  # "autoregression" #"non_linear_autoregression"  # "autoregression"
        self.n_estimator = 10
        self.base_regressor = firets_base_regressor  # XGBRegressor()
        self.exp_name = ""
        # self.base_regressor = RandomForestRegressor()

        # experiment parameter
        self.verbose = 2
        self.save_model_path = None  # "output/Base_lstm_" + self.target[0]


class StatParams:
    def __init__(self, model_name, output_time_steps, holiday_features, target=None, exog_lst=None, cut=400):
        self.model_name = model_name
        self.target = target
        self.exog_lst = exog_lst  #["RelativeHumidity"]  # ["WindSpeed10"]
        self.cut = cut
        self.output_time_steps = output_time_steps
        self.holiday_features = holiday_features
