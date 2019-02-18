"""
Given a model architecture, write it to a queue to have data tested

Define the model in create_model() and then call

`docker-compose run --rm model_listener`

That will generate a json representation of the model and write it to the queue
"""
import numpy as np
import json
import keras
import common as co
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

QUEUE_MODEL_NAME="footballModelsArchitectures"
QUEUE_MODEL_RESULTS_NAME="footballModelsArchitectureResults"
# FIXME: this should be shared across most of these classes
n_classes = 2

def create_model_old():
    """Any model created here will be json'ified"""
    model = Sequential()
    # conv 1
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     #FIXME: extract this shape into a system-wide variable
                     input_shape=(240, 320, 1),
                     name="conv_1_1"))

    # model.add(Conv2D(32,
    #                  kernel_size=(3, 3),
    #                  activation='relu',
    #                  padding='same',
    #                  #FIXME: extract this shape into a system-wide variable
    #                  name="conv_1_2"))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same',
                           name="max_pool_1"))
    # model.add(Dropout(0.05))

    # conv 2
    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name="conv_2_1"))

    # model.add(Conv2D(64,
    #                  kernel_size=(3, 3),
    #                  activation='relu',
    #                  padding='same',
    #                  name="conv_2_2"))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same',
                           name="max_pool_2"))

    model.add(Conv2D(128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name="conv_3_1"))


    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='same',
                           name="max_pool_3"))

    model.add(Dropout(0.05))

    # Fully Connected
    model.add(Flatten())

    model.add(Dense(2048, activation='relu', name="fc_1"))
    model.add(Dropout(0.1))

    model.add(Dense(n_classes, activation='softmax', name="output"))
    return model

def create_model():
    """Any model created here will be json'ified"""
    model = Sequential()
    # conv 1
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     #FIXME: extract this shape into a system-wide variable
                     input_shape=(240, 320, 1),
                     name="conv_1_1"))

    # Fully Connected
    model.add(Flatten())

    model.add(Dense(48, activation='relu', name="fc_1"))

    model.add(Dense(n_classes, activation='softmax', name="output"))
    return model

def write_to_consumer(model):
    """write a keras model to a kafka queue"""

    co.MODEL_PRODUCER.send(QUEUE_MODEL_NAME,
                           model.to_json().encode('utf-8'))
    # NOTE: this flush is required to send it right away
    co.MODEL_PRODUCER.flush()

def main():
    model = create_model()
    write_to_consumer(model)

if __name__ == "__main__":
    print("generating model")
    main()
    print("done")
