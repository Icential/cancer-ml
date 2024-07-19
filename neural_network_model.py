
# classification neural network algorithm for predicting cancer
# using keras

# now: implement validation testing
# later: improve model, test feature crossing
# known issues: fix model training x features


# imports here
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignores tf warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



"""Setting up datasets"""
# put csv into a pd dataframe
df = pd.read_csv("cancer.csv")

# split into training and test sets
train, test = train_test_split(df, test_size=.2)



"""Create model"""

def create_inputs(features):
    inputs = {}
    input_layers = []

    # create dictionary for each chosen feature's input layer
    for i in range(len(features)):
        inputs.update({features[i]: tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name=features[i])})
    
    # normalize each input layer
    for i in range(len(features)):
        filler = tf.keras.layers.Normalization(name=features[i], axis=None)
        filler.adapt(train[features[i]])
        filler = filler(inputs.get(features[i]))
        input_layers.append(filler)

    # concatenate all input layers
    preprocessing = tf.keras.layers.Concatenate()(input_layers)

    return preprocessing

def get_outputs(preprocessing):

    # create each layer
    layer_outputs = tf.keras.layers.Dense(units=8,
                                   activation='relu')(preprocessing)
    layer_outputs = tf.keras.layers.Dense(units=4,
                                   activation='relu')(layer_outputs)
    layer_outputs = tf.keras.layers.Dense(units=1,
                                    activation='sigmoid')(layer_outputs)
    
    # put in dictionary form
    outputs = {
        'layer_outputs': layer_outputs
    }

    return outputs

def build_model(inputs, outputs, learning_rate):

    # combine inputs and outputs to create a model 
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # compile model with chosen loss, optimizer and metrics
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])

    return model



""""Train model"""
def train_model(model, dataset, epochs, batch_size, label_name):

    # fit model according to chosen epoch and batch size
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    results = model.fit(x=features, y=label, epochs=epochs, batch_size=batch_size, shuffle=True)

    # get epoch
    result_epochs = results.epoch

    # isolte epoch and get other results
    hist = pd.DataFrame(results.history)

    return result_epochs, hist



"""Evaluate model"""
# plot evaluations over epochs of the model  
def plot_loss(epochs, hist):
    
    # label axes
    plt.xlabel('Epoch')
    plt.ylabel('Evaluations')

    # plot evaluations in respect to epochs
    plt.plot(epochs, hist['accuracy'][1:], label='Accuracy')
    plt.legend()
    plt.show()



"""Testing Playground"""
# labels and features
features = [
    "ALLERGY",
    "ALCOHOL_CONSUMING"
]
label_name = "LUNG_CANCER"


# hyperparameters
learning_rate = 0.01
epochs = 50
batch_size = 10
binary_threshold = 0.6


# feature crossing
inputs = create_inputs(features)
outputs = get_outputs(inputs)

# model creation with above input-outputs
model = build_model(inputs, outputs, learning_rate)
epochs, hist = train_model(model, train, epochs, batch_size, label_name)
plot_loss(epochs, hist)

print("\n")