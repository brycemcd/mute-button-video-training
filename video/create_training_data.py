"""
Expects a vector of training data on a kafka queue, create training/test data
"""

import numpy as np
from kafka import KafkaConsumer
import json

QUEUE=KafkaConsumer('supervised_vectorized_images',
    bootstrap_servers='10.1.2.206:9092',
    value_deserializer=lambda v: json.loads(v),
    # group_id='my_favorite_group1',
    auto_offset_reset='earliest',
)

IMG_SHAPE = [1, 240, 320, 1]
DATA_SHAPE = [76800]
TEST_SAMPLE_COUNT=1000

# NOTE: /tmp/bin is mapped to the host filesystem in docker
def save_training_data(filepath="/tmp/bin"):

    samples, labels = create_training_test_data()

    np.save(filepath + "/normalized_all_samples", samples)
    np.save(filepath + "/all_labels", labels)

def create_data_split(samples, labels):
    X_train, X_test, y_train, y_test = train_test_split(samples, labels,
                                                        test_size=0.25,
                                                        )
    return X_train, X_test, y_train, y_test

def create_training_test_data():
    """docstring for create_training_test_data"""
    SAMPLES=np.zeros(IMG_SHAPE)
    LABELS=np.zeros([1, 2]) # 1 , num_classes

    for msg in consume_queue():
        label = msg[0]
        sample = msg[1:]

        SAMPLES = np.vstack((SAMPLES,
                             np.array(sample).reshape(IMG_SHAPE)))
        if label == 0:
            label = [0, 1]
        else:
            label = [1, 0]

        LABELS = np.vstack((LABELS,
                            np.array(label)))

        if len(SAMPLES) >= TEST_SAMPLE_COUNT:
            break
    SAMPLES -= np.mean([SAMPLES], axis=1)
    SAMPLES /= 255 # normalize to the max value
    return SAMPLES, LABELS

def consume_queue():
    """docstring for consume_queue"""
    for msg in QUEUE:
        yield(msg.value['sample'])


if __name__ == "__main__":
    save_training_data()
