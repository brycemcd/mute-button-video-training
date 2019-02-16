"""
Expects a vector of training data on a kafka queue. Creates training data with
labels and persists the data to the filesystem.

NOTE: on my 32GB machine, having > ~ 100000 training samples starts to create
some pressure on memory/swap space. The type casts in `create_training_data`
are to reduce this pressure in order to optimize getting the most data available
for training
"""

import numpy as np
import common as co

# NOTE: run scratch_1 to figure out the last offset
LAST_OFFSET = 22000


# NOTE: /tmp/bin is mapped to the host filesystem in docker
def save_training_data(filepath="/tmp/models_and_training_data"):

    samples, labels = create_training_data()

    print("saving all samples")
    np.save(filepath + "/normalized_all_samples", samples)
    print("saving all labels")
    np.save(filepath + "/all_labels", labels)


def create_training_data():
    """docstring for create_training_data"""
    # setting the type for known memory constraints vs. amount of data tradeoff
    single_image_vec_shape = [240, 320, 1]
    samples=np.empty(co.IMG_SHAPE, dtype=np.uint8)
    labels=np.empty([co.TEST_SAMPLE_COUNT, 2], dtype=np.uint8) # n , num_classes

    i = 0
    for msg in consume_queue():
        jmsg = msg.value['sample']
        label = jmsg[0]
        sample = jmsg[1:]

        samples[i] = np.array(sample, dtype=np.uint8).reshape(single_image_vec_shape)

        if label == co.GAME_LABEL:
            label = co.GAME_VECTOR
        else:
            label = co.NOT_GAME_VECTOR

        labels[i] = np.array(label, dtype=np.uint8)

        i += 1
        # NOTE: read the queue in its entirety
        # if msg.offset >= LAST_OFFSET:
        if i >= samples.shape[0]:
            break

        if i % 1000 == 0:
            print(".", end='', sep='', flush=True)

    # NORMALIZE SAMPLES:
    # Helpful background: http://cs231n.github.io/neural-networks-2/#datapre
    # NOTE: if you get this warning: RuntimeWarning: overflow encountered in reduce
    # then the centering is infinity and your training data are crap
    print("CENTERING")
    center = np.mean(samples)
    print("CENTER IS %s. ADD THIS TO classify_image.py for prediction treatment" % center)
    # NOTE: This doubles the amount of memory needed for this op but casts it as
    # the correct type
    final_samples=np.empty(co.IMG_SHAPE, dtype=np.int8)
    for i in range(samples.shape[0]):
        final_samples[i] = co.center_sample(samples[i], center)

    return final_samples, labels


def consume_queue():
    """docstring for consume_queue"""
    for msg in co.VECTORED_IMAGE_QUEUE:
        yield(msg)


def main():
    print("saving")
    save_training_data()
    print("done!")


if __name__ == "__main__":
    main()
