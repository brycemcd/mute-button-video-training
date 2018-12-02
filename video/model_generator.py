"""
Creates models for training. See:
https://github.com/brycemcd/learning_neural_networks/blob/master/training-chain/2017-11-23-model-creator.ipynb
For my scratch work on this.

Model is saved to the filesystem to be used in predictions.
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

# FIXME: this should be shared across most of these classes
n_classes = 2

def create_model():
    """docstring for create_model"""
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


def train_model(train_x, train_y):
    x_train, x_valid, x_train_labels, x_valid_labels = create_data_split(train_x, train_y)
    model = create_model()

    # TODO: use some sort of grid search to optimize this
    sgd = keras.optimizers.SGD(lr=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # NOTE: training speed takes a lot of guess and check on the system
    # the training is happening. I regularly run `vmstat 2` to keep an eye
    # on memory, disk and CPU usage. Memory can explode very quickly. Adjust
    # batch size to prevent swapping and to keep the amount of resident memory
    # close to a manageable allocation for the system.
    #
    # Ideally, the system is tuned such that we can bet the stuffing out of the
    # (C|G)PUs while optimizing the dataset that can be held resident in RAM
    training_hx = model.fit(
        x_train,
        x_train_labels,
        batch_size=128, #220
        epochs=20, #FIXME: turn this back into 10
        verbose=1,
        validation_data=(x_valid, x_valid_labels))

    return training_hx, model


def test_model(model, test_x, test_y):
    return model.evaluate(test_x, test_y, batch_size=128, verbose=1)


def save_model(model, filepath="/tmp/models_and_training_data"):
    model.save(filepath + '/football_cnn.h5')


def create_data_split(samples, labels):
    return train_test_split(samples, labels, test_size=0.20)


def load_data_from_filesystem(basepath="/tmp/models_and_training_data"):
    samples = np.load(basepath + "/normalized_all_samples.npy")
    labels = np.load(basepath + "/all_labels.npy")
    return samples, labels


def main():
    s, l = load_data_from_filesystem()

    x_train, x_test, y_train, y_test = create_data_split(s, l)

    # From X_train numbers, get 3000 indexes
    # random_indexes = np.random.choice(len(x_train), 3000)
    # FIXME: time optimization:
    train_hx, model = train_model(x_train, y_train)
    training_eval = test_model(model, x_test, y_test)

    print("TEST SET PERF: %s" % dict(zip(model.metrics_names, training_eval)))

    save_model(model)


if __name__ == "__main__":
    main()
