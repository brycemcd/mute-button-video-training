"""
Listens to queue to receive model architecture and trains model

TODO: requirements for data
"""

import json
import socket
from datetime import datetime as dt
import numpy as np
import common as co
import keras
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

QUEUE_MODEL_NAME="footballModelsArchitectures"
QUEUE_MODEL_RESULTS_NAME="footballModelsArchitectureResults"
# FIXME: this should be shared across most of these classes
n_classes = 2

def train_model(train_x, train_y, model):
    x_train, x_valid, x_train_labels, x_valid_labels = create_data_split(train_x, train_y)
    # model = create_model()

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
    """saves model to filesystem"""

    now = dt.now().isoformat(timespec='seconds')
    model.save(filepath + "/football_cnn-" + now + ".h5")


def create_data_split(samples, labels):
    return train_test_split(samples, labels, test_size=0.20)


def load_data_from_filesystem(basepath="/tmp/models_and_training_data"):
    samples = np.load(basepath + "/normalized_all_samples.npy")
    labels = np.load(basepath + "/all_labels.npy")
    return samples, labels


def publish_results(result_dict):
    """Publishes results to a queue"""

    result_dict['hostname'] = socket.gethostname()
    result_dict['dttm'] = dt.now().isoformat()

    co.PRODUCER.send(QUEUE_MODEL_RESULTS_NAME, result_dict)
    co.PRODUCER.flush()

def main():
    """docstring for consume_queue"""
    print("starting up")
    s, l = load_data_from_filesystem()

    x_train, x_test, y_train, y_test = create_data_split(s, l)

    print("data loaded, ready for models")
    for msg in co.MODEL_CONSUMER:
        model = model_from_json(json.dumps(msg.value))
        # From X_train numbers, get 3000 indexes
        # random_indexes = np.random.choice(len(x_train), 3000)
        # FIXME: time optimization:
        train_hx, model = train_model(x_train[1:10], y_train[1:10], model)
        training_eval = test_model(model, x_test[1:10], y_test[1:10])

        results = dict(zip(model.metrics_names, training_eval))
        publish_results(results)
        print("TEST SET PERF: %s" % results)
        save_model(model)


if __name__ == "__main__":
    main()
