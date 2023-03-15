
# classification neural network algorithm for predicting cancer
# using keras and feature crossing (up to 95% accuracy)

# now: implement feature crosses 
# later: confusion matrix, roc and auc


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

# split into training and test sets
train, test = train_test_split(df, test_size=.2)



"""Create model"""
def build_model(learning_rate, feature_layer):

    # model that i made up with four layers
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=16, activation='sigmoid', input_shape=(1,)))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # define chosen optimizer and learning rate
    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    # compile model with chosen loss, optimizer and metrics
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model



""""Train model"""
def train_model(model, dataset, epochs, batch_size, label_name):

    # fit model according to chosen epoch and batch size
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    results = model.fit(features, label, epochs=epochs, batch_size=batch_size, shuffle=True)

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
def model_accuracy(model, feature_name_one, feature_name_two, label_name):

    # get predictions (probabilities) with x_test (unbiased to trained set)
    x_test = test[feature_name_one].combine(test[feature_name_two], np.minimum).values.tolist()
    y_test = test[label_name].values.tolist()

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


# feature crossing
chosen_features = []
alcohol_consumption = tf.feature_column.numeric_column('ALCOHOL_CONSUMING')
chosen_features.append(alcohol_consumption)
allergy = tf.feature_column.numeric_column('ALLERGY')
chosen_features.append(allergy)
feature_layer = tf.keras.layers.DenseFeatures(chosen_features)


model = build_model(learning_rate, feature_layer)
trained_weight, trained_bias, epochs, rmse = train_model(model, train, epochs, batch_size, label_name)
plot_loss(epochs, rmse)
# model_accuracy(model, feature_name_one, feature_name_two, label_name)

print("\n")