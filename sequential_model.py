
# classification neural network algorithm for predicting cancer
# using keras and feature crossing (up to 93% accuracy)


# imports here
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignores tf warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



"""Setting up datasets"""
# put csv into a pd dataframe
df = pd.read_csv("cancer.csv")

x = df["ALLERGY"]
y = df["LUNG_CANCER"]

# split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)



"""Create model"""
def build_model(learning_rate):

    # model that i made up with four layers
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=16, activation='sigmoid', input_shape=(1,)))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # define chosen optimizer and learning rate
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # compile model with chosen loss, optimizer and metrics
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model



""""Train model"""
def train_model(model, epochs, batch_size):

    # fit model according to chosen epoch and batch size
    results = model.fit(x_test, y_test, epochs=epochs, batch_size=batch_size, shuffle=True)

    # get weights and bias
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # get epoch
    epochs = results.epoch

    # isolte epoch and get other results
    new_results = pd.DataFrame(results.history)

    # get root mean squared
    rmse = new_results['root_mean_squared_error']

    return trained_weight, trained_bias, epochs, rmse



"""Evaluate model"""
# plot loss (rmse) over epochs of the model  
def plot_loss(epochs, rmse):
    
    # label axes
    plt.xlabel('Epoch')
    plt.ylabel('Root Mean Squared Error')

    # plot rmse in respect to epochs
    plt.plot(epochs, rmse, label='Loss')
    plt.legend()
    plt.ylim([rmse.min()*0.95, rmse.max()])
    plt.show()


# calculate model accuracy
def model_accuracy(model):

    # predict with test set
    y_hat = model.predict(x_test)

    # normalize to 0s and 1s
    y_hat = [0 if val < 0.5 else 1 for val in y_hat]

    # overall accuracy in comparison with the real dataset (y test for no bias to trained set)
    print('\nAverage overall accuracy of', accuracy_score(y_hat, y_test))

    model.evaluate(x=x_test, y=y_test, batch_size=25)



"""Testing Playground"""
# labels and features
feature_name_one = "ALLERGY"
feature_name_two = "ALCOHOL_CONSUMING"
label_name = "LUNG_CANCER"


# hyperparameters
learning_rate = 0.01
epochs = 50
batch_size = 4


model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(model, epochs, batch_size)
plot_loss(epochs, rmse)
model_accuracy(model)

print("\n")