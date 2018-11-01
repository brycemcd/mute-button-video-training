"""
Expects a vector of training data on a kafka queue, create training/test data
"""

import numpy as np
from kafka import KafkaConsumer
import json

QUEUE=KafkaConsumer('supervised_vectorized_images',
# QUEUE=KafkaConsumer('unsupervised_images',
    bootstrap_servers='10.1.2.206:9092',
    # NOTE: use this for sample, not encoded data
    value_deserializer=lambda v: json.loads(v),
    # group_id='my_favorite_group1',
    auto_offset_reset='earliest',
)

# TEST_SAMPLE_COUNT=189000
# TEST_SAMPLE_COUNT=81500 # NOTE: this is the max my machine's memory can tollerate
TEST_SAMPLE_COUNT=90000 # NOTE: this is the max my machine's memory can tollerate
# TEST_SAMPLE_COUNT=100
IMG_SHAPE = [TEST_SAMPLE_COUNT, 240, 320, 1]
DATA_SHAPE = [76800]


# NOTE: /tmp/bin is mapped to the host filesystem in docker
def save_training_data(filepath="/tmp/models_and_training_data"):

    # samples, labels = create_training_test_data()
    samples, labels = memory_test()

    print("saving all samples")
    np.save(filepath + "/normalized_all_samples", samples)
    print("saving all labels")
    np.save(filepath + "/all_labels", labels)


def create_data_split(samples, labels):
    X_train, X_test, y_train, y_test = train_test_split(samples, labels,
                                                        test_size=0.25,
                                                        )
    return X_train, X_test, y_train, y_test


def memory_test():
    single_image_vec_shape = [240, 320, 1]
    samples=np.empty(IMG_SHAPE, dtype=np.uint8)
    labels=np.empty([TEST_SAMPLE_COUNT, 2], dtype=np.uint8) # n , num_classes

    i = 0
    for _ in range(TEST_SAMPLE_COUNT):
        label = [0, 1]
        sample = np.random.rand(76800) * 100

        samples[i] = np.array(sample, dtype=np.uint8).reshape(single_image_vec_shape)

        labels[i] = np.array(label, dtype=np.uint8)

        i += 1
        if i % 1000 == 0:
            print("i = %s" % i)
        # print(".", end='', sep='', flush=True)
        # if len(samples) >= TEST_SAMPLE_COUNT:
        if i >= TEST_SAMPLE_COUNT:
            break

    # NORMALIZE SAMPLES:
    center = np.mean(samples, dtype=np.float16)
    # NOTE: memory claim with 100000 rows
    # normed_samples=np.empty(IMG_SHAPE, dtype=np.int8) # occupies ~ 7.6 GB of memory
    # normed_samples=np.empty(IMG_SHAPE, dtype=np.float16) # occupies ~ 15.3 GB of memory
    # normed_samples = (samples - np.mean([samples], axis=1) * 100 // 255).astype(np.int8)
    # normed_samples /= 255 # normalize to the max value

    for i in range(samples.shape[0]):
        row = samples[i].astype(np.float16)
        # NOTE: if training speed is low, then normalize by uncommenting:
        # row /= 255
        row -= center

        # NOTE: casting this back to an int has the effect of rounding
        # We lose precision but gain memory space (8 vs. 16 or 32 bits per value)
        samples[i] = row.astype(np.uint8)


    return samples, labels

def create_training_test_data():
    """docstring for create_training_test_data"""
    # setting the type for known memory constraints vs. amount of data tradeoff
    single_image_vec_shape = [240, 320, 1]
    samples=np.empty(IMG_SHAPE, dtype=np.uint8)
    labels=np.empty([TEST_SAMPLE_COUNT, 2], dtype=np.uint8) # n , num_classes

    i = 0
    for msg in consume_queue():
        label = msg[0]
        sample = msg[1:]

        # FIXME: this is really really slow. Reallocating array
        # samples = np.vstack((samples,
                             # np.array(sample).reshape(IMG_SHAPE)))
        # NOTE: these are all values between 0 and 255 right now
        samples[i] = np.array(sample, dtype=np.uint8).reshape(single_image_vec_shape)

        if label == 0:
            label = [0, 1]
        else:
            label = [1, 0]

        labels[i] = np.array(label, dtype=np.uint8)
        # labels = np.vstack((labels,
                            # np.array(label)))

        print(".", end='', sep='', flush=True)
        # if len(samples) >= TEST_SAMPLE_COUNT:
        i += 1
        if i >= TEST_SAMPLE_COUNT:
            break

    normed_samples = samples - np.mean([samples], axis=1)
    normed_samples /= 255 # normalize to the max value
    return samples, labels


def consume_queue():
    """docstring for consume_queue"""
    for msg in QUEUE:
        yield(msg.value['sample'])


if __name__ == "__main__":
    print("saving")
    save_training_data()
    print("done!")
