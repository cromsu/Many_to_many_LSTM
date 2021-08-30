import timeit
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Dropout


class Neuralnet:
    '''
    Neural network class. Contains the methods that use tf or keras resp.
    
    Class variables
    ---------------
    model : None type
        Sequential Keras model.
        
    Methods
    -------
    define
    train
    feedforward
    
    '''
    def define():
        '''
        Makes the model class variable.
        model : LSTM + Dense

        Returns
        -------
        None.

        '''
        hidden_neurons = int(input('hidden_neurons (int; x>0):'))

        Neuralnet.model = Sequential()
        Neuralnet.model.add(LSTM(hidden_neurons, return_sequences=True))
        Neuralnet.model.add(Dropout(0.2))
        Neuralnet.model.add(LSTM(hidden_neurons, return_sequences=True))
        Neuralnet.model.add(TimeDistributed(Dense(1)))

        opt = tf.keras.optimizers.Adam()
        Neuralnet.model.compile(loss='mean_squared_error', optimizer=opt)

        return None

    def train(x_train, y_train):
        '''
        trains Neuralnet.model .

        Parameters
        ----------
        x_train : array, float
            x train Data.
        y_train : array, float
            y train data.

        Returns
        -------
        None.

        '''
        epo = int(input('epochs (int; x>0):'))
        time_start = timeit.default_timer()
        Neuralnet.model.fit(x_train, y_train, epochs = epo)
        time_finished = timeit.default_timer()

        print("Done learning. Time:", time_finished - time_start, "seconds")

        return None

    def feedforward(seq_lenght):
        '''
        feedforward Neuralnet.model for a number of times steps given by a 
        console promt.

        Parameters
        ----------
        seq_lenght : array, float
            Train data arrays lenght.

        Returns
        -------
        y : array, float
            Models prediction.

        '''
        test_lenght = int(input('timestapes to predict (int; x>=0):'))
        x = np.linspace(0, seq_lenght+test_lenght-1, seq_lenght+test_lenght)
        x = x.reshape(1, len(x), 1)
        y = Neuralnet.model(x)

        return y