from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM, TimeDistributed, Flatten
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.layers import Layer


class BaseLSTM:
    """
    this class builds a LSTM model for AQI prediction

    input:
    ------
    hidden_dim: hyper-parameter, dimension of the hidden layer in LSTM Network
    input_shape: size of the input feature vector
    output_shape: size of the prediction step
    kernel: a string which specifies the kernel of the RNN Network (LSTM, GRU, SimpleRNN)
    input_dropout: dropout rate of the input layer
    recurrent_dropout: dropout rate of the hidden layer
    stacket_layer_num: number of the hidden layers in RNN network
    loss: specifies the loss function to train RNN network
    optimizer: the optimization method, default is "adam"
    add_attention: a parameter to add an attention layer
    """
    def __init__(self, hidden_dim, input_shape, output_shape, kernel,
                 input_dropout, recurrent_dropout, stacket_layer_num, loss,
                 optimizer='adam', add_attention=True):
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel = kernel
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout
        self.stacket_layer_num = stacket_layer_num
        self.loss = loss
        self.optimizer = optimizer
        self.add_attention = add_attention

        self.__model = self.__build_network()

    def __build_network(self):
        # design network
        model = Sequential()
        if self.kernel == "GRU":
            for i in range(self.stacket_layer_num):
                model.add(GRU(self.hidden_dim, input_shape=self.input_shape, return_sequences=True,
                              dropout=self.input_dropout, recurrent_dropout=self.recurrent_dropout))
            model.add(GRU(self.hidden_dim, input_shape=self.input_shape,
                          dropout=self.input_dropout, recurrent_dropout=self.recurrent_dropout))
        elif self.kernel == "SimpleRNN":
            for i in range(self.stacket_layer_num):
                model.add(SimpleRNN(self.hidden_dim, input_shape=self.input_shape, return_sequences=True,
                                    dropout=self.input_dropout, recurrent_dropout=self.recurrent_dropout))
            model.add(SimpleRNN(self.hidden_dim, input_shape=self.input_shape,
                                dropout=self.input_dropout, recurrent_dropout=self.recurrent_dropout))
        else:
            for i in range(self.stacket_layer_num):
                model.add(LSTM(self.hidden_dim, input_shape=self.input_shape, return_sequences=True,
                               dropout=self.input_dropout, recurrent_dropout=self.recurrent_dropout))

            model.add(LSTM(self.hidden_dim, input_shape=self.input_shape,
                           dropout=self.input_dropout, recurrent_dropout=self.recurrent_dropout))

        model.add(Dense(self.output_shape))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        # print(model.summary())
        return model

    def fit(self, x, y, epochs, batch_size, validation_data, verbose, shuffle=False):
        # fit network
        history = self.__model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                   validation_data=validation_data, verbose=verbose,
                                   shuffle=shuffle)
        return history

    def predict(self, x_test):
        yhat = self.__model.predict(x_test)
        return yhat

    def save(self, path="model.h5"):
        self.__model.save(path)

    def restore(self, path='model.h5'):
        self.__model = load_model(path)


class BaseCNN:
    """
    this class builds a CNN model for AQI prediction

    input:
    ------
    filters: a list specifying the filters
    kernel_size: a list containing the size of the kernels
    activation: specifies the activation function of the network
    input_shape: size of the input feature vector
    output_shape: size of the prediction step
    dense_layer_size: size of the fully connected layers
    loss: specifies the loss function to train RNN network
    optimizer: the optimization method, default is "adam"
    add_attention: a parameter to add an attention layer
    """
    def __init__(self, filters, kernel_size, activation,
                 input_shape, output_shape, pool_size, dense_layer_size,
                 loss, optimizer='adam'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pool_size = pool_size
        self.dense_layer_size = dense_layer_size
        self.loss = loss
        self.optimizer = optimizer
        self.__model = self.__build_network()

    def __build_network(self):
        # design network
        model = Sequential()
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                         activation=self.activation, input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(Flatten())
        model.add(Dense(self.dense_layer_size, activation=self.activation))
        model.add(Dense(self.output_shape))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        # print(model.summary())
        return model

    def fit(self, x, y, epochs, batch_size, validation_data, verbose, shuffle=False):
        # fit network
        history = self.__model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                   validation_data=validation_data, verbose=verbose,
                                   shuffle=shuffle)
        return history

    def predict(self, x_test):
        yhat = self.__model.predict(x_test)
        return yhat

    def save(self, path="model.h5"):
        self.__model.save(path)

    def restore(self, path='model.h5'):
        self.__model = load_model(path)


class CnnLSTM:
    """
    this class builds a CnnLSTM model for AQI prediction

    input:
    ------
    filters: a list specifying the filters
    kernel_size: a list containing the size of the kernels
    activation: specifies the activation function of the network
    hidden_dim: hyper-parameter, dimension of the hidden layer in LSTM Network
    input_shape: size of the input feature vector
    output_shape: size of the prediction step
    input_dropout: dropout rate of the input layer
    recurrent_dropout: dropout rate of the hidden layer
    stacket_layer_num: number of the hidden layers in RNN network
    loss: specifies the loss function to train RNN network
    optimizer: the optimization method, default is "adam"
    """
    def __init__(self, filters, kernel_size, activation,
                 hidden_dim, input_shape, output_shape, pool_size,
                 input_dropout, recurrent_dropout, stacket_layer_num, loss,
                 optimizer='adam'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pool_size = pool_size

        self.hidden_dim = hidden_dim
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout
        self.stacket_layer_num = stacket_layer_num
        self.loss = loss
        self.optimizer = optimizer

        self.__model = self.__build_network()

    def __build_network(self):
        # design network
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                                         activation=self.activation),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(MaxPooling1D(pool_size=self.pool_size)))
        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(self.hidden_dim, activation=self.activation))
        model.add(Dense(self.output_shape))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        # print(model.summary())
        return model

    def fit(self, x, y, epochs, batch_size, validation_data, verbose, shuffle=False):
        # fit network
        history = self.__model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                   validation_data=validation_data, verbose=verbose,
                                   shuffle=shuffle)
        return history

    def predict(self, x_test):
        yhat = self.__model.predict(x_test)
        return yhat

    def save(self, path="model.h5"):
        self.__model.save(path)

    def restore(self, path='model.h5'):
        self.__model = load_model(path)


class BaseMLP:
    def __init__(self, hidden_dim, input_dim, output_dim, activation, loss='mae', optimizer='adam'):
        """
            hidden_dim: a list containing hidden dimensions of MLP network. the length of this
                list is equal to the number of hidden layers in MLP network
            input_dim: An integer specifying the input dimension of each sample
            output_dim: An integer specifying  the output dimension of each sample
        """
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.__model = self.__build_network()

    def __build_network(self):
        # define model
        model = Sequential()
        prev_dim = self.input_dim
        for dim in self.hidden_dim:
            model.add(Dense(dim, activation=self.activation, input_dim=prev_dim))
            prev_dim = dim
        model.add(Dense(self.output_dim))
        model.compile(optimizer='adam', loss='mse')
        print(model.summary())
        return model

    def fit(self, x, y, epochs, batch_size, validation_data, verbose, shuffle=False):
        # fit network
        history = self.__model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                   validation_data=validation_data, verbose=verbose,
                                   shuffle=shuffle)
        # model.fit(X, y, epochs=2000, verbose=0)
        return history

    def predict(self, x_test):
        yhat = self.__model.predict(x_test)
        return yhat

    def save(self, path="model.h5"):
        self.__model.save(path)

    def restore(self, path='model.h5'):
        self.__model = load_model(path)


class MultiHeadedMLP:
    def __init__(self, hidden_dim, head_dim, output_dim, loss='mae', optimizer='adam'):
        """
            hidden_dim: a list containing hidden dimensions of MLP network. the length of this
                list is equal to the number of hidden layers in MLP networ
            head_dim: A list containing the dimension of each head in MLP
            output_dim: An integer specifying  the output dimension of each sample
        """
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.output_dim = output_dim
        self.loss = loss
        self.optimizer = optimizer
        self.__model = self.__build_network()

    def __build_network(self):
        # define model

        input_list = []
        head_list = []
        for head in self.head_dim:
            # head input model
            visible = Input(shape=(head,))
            dense = Dense(self.hidden_dim, activation='relu')(visible)
            input_list.append(visible)
            head_list.append(dense)
        # merge input models
        merge = concatenate(head_list)
        output = Dense(1)(merge)
        model = Model(inputs=input_list, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        print(model.summary())

        return model

    def fit(self, x, y, epochs, batch_size, validation_data, verbose, shuffle=False):
        # fit network
        history = self.__model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                   validation_data=validation_data, verbose=verbose,
                                   shuffle=shuffle)
        # model.fit(X, y, epochs=2000, verbose=0)
        return history

    def predict(self, x_test):
        yhat = self.__model.predict(x_test)
        return yhat

    def save(self, path="model.h5"):
        self.__model.save(path)

    def restore(self, path='model.h5'):
        self.__model = load_model(path)


class BaseAttention:
    """
    this class builds a LSTM model for AQI prediction

    input:
    ------
    hidden_dim: hyper-parameter, dimension of the hidden layer in LSTM Network
    input_shape: size of the input feature vector
    output_shape: size of the prediction step
    kernel: a string which specifies the kernel of the RNN Network (LSTM, GRU, SimpleRNN)
    input_dropout: dropout rate of the input layer
    recurrent_dropout: dropout rate of the hidden layer
    stacket_layer_num: number of the hidden layers in RNN network
    loss: specifies the loss function to train RNN network
    optimizer: the optimization method, default is "adam"
    """
    def __init__(self, hidden_dim, input_shape, output_shape, kernel,
                 input_dropout, recurrent_dropout, stacket_layer_num, loss, optimizer='adam'):
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel = kernel
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout
        self.stacket_layer_num = stacket_layer_num
        self.loss = loss
        self.optimizer = optimizer

        self.__model = self.__build_network()

    def __build_network(self):
        # design network
        inputs = Input(shape=self.input_shape)
        x = LSTM(self.hidden_dim, input_shape=self.input_shape,
                 dropout=self.input_dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True)(inputs)
        hidden_states = x
        hidden_size = int(hidden_states.shape[2])
        tmp = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([tmp, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)

        x = Dense(self.output_shape)(attention_vector)
        model = Model(inputs=inputs, outputs=x, name="mnist_model")

        model.compile(loss=self.loss, optimizer=self.optimizer)
        print(model.summary())
        return model

    def fit(self, x, y, epochs, batch_size, validation_data, verbose, shuffle=False):
        # fit network
        history = self.__model.fit(x, y, epochs=epochs, batch_size=batch_size,
                                   validation_data=validation_data, verbose=verbose,
                                   shuffle=shuffle)
        return history

    def predict(self, x_test):
        yhat = self.__model.predict(x_test)
        return yhat

    def save(self, path="model.h5"):
        self.__model.save(path)

    def restore(self, path='model.h5'):
        self.__model = load_model(path)
