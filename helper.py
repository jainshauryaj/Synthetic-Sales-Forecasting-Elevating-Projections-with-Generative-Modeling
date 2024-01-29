import pandas as pd
import numpy as np
from datetime import date
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras import losses
from keras.layers import Layer, PReLU, BatchNormalization, Dropout, ReLU, LSTM, Input, Dense, Lambda
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras import backend as K

class GradientCheckCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        # Get the current model
        model = self.model

        # List to store the gradients
        gradients = []

        # Iterate over the layers
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                # Get the gradients of the trainable weights
                grads = K.gradients(model.total_loss, layer.trainable_weights)
                evaluated_gradients = K.eval(grads[0])
                gradients.append(evaluated_gradients)

        # Check if gradients are too small (vanishing) or too large (exploding)
        for gradient in gradients:
            if np.max(gradient) > 1e5:
                print('Gradient explosion detected')
            if np.max(gradient) < 1e-5:
                print('Gradient vanishing detected')
            else:
                print('it is fine')

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VaeLossLayer(Layer):
    def vae_loss(self, x, x_decoded_mean, z_mean, z_log_var):
        xent_loss = losses.mean_squared_error(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x

def build_vae_model():
    # Encoder
    encoder_inputs = Input(shape=(input_dim,))
    x = Dense(256, activation="relu")(encoder_inputs)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(32, activation="relu")(latent_inputs)
    x = Dense(64, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    decoder_outputs = Dense(input_dim, activation="linear")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    # Variational Autoencoder
    encoder_outputs = encoder(encoder_inputs)
    vae_outputs = decoder(encoder_outputs[2])
    y = VaeLossLayer()([encoder_inputs, vae_outputs, encoder_outputs[0], encoder_outputs[1]])
    vae = Model(encoder_inputs, y, name="vae")

    return vae

def get_seasonal_periods(check):
    """
    input : Pass the time series

    return : List of seasonal periods
    """

    # Apply Fourier Transform
    f = np.fft.fft(check)
    frequencies = np.fft.fftfreq(len(check))

    # Get absolute values and sort them
    magnitudes = np.abs(f)
    sorted_indices = np.argsort(magnitudes)[::-1]

    # Get seasonalities
    seasonalities = frequencies[sorted_indices]

    # Filter seasonalities
    seasonal_perdiods = [int(i) for i in seasonalities if (i < int(0.8*len(check))) & (i > 2) & (i < 100)]
    seasonal_perdiods.append(7)
    seasonal_perdiods.append(12)
    seasonal_perdiods.append(52)
    return list(set(seasonal_perdiods))

from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.metrics import make_scorer

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def tune_HW(check, seasonal_periods):
    # Divide into an 80/20 training-test split
    split = int(0.8*len(check))
    train_ts = check[0:split]
    test_ts = check[split:]

    # Define the model
    model = ExponentialSmoothing(train_ts)

    # Define the grid of parameters to search over
    parameters_grid_search = {
        "trend": ['mul', 'add'],
        "seasonal": ['mul', 'add'],
        "seasonal_periods": seasonal_periods
    }

    # Define the scoring function
    scoring = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False)
    }

    # Create the grid search object
    grid = GridSearchCV(model, parameters_grid_search, scoring=scoring, refit='mape')

    # Fit the grid search object to the data
    grid.fit(train_ts, test_ts)

    # Get the best parameters and their corresponding score
    tune_parm = grid.best_params_
    tune_mape = grid.best_score_

    return tune_parm, tune_mape

