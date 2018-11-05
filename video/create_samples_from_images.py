"""
Given a directory of images, this script will base64 encode the image and write
it to a Kafka queue

TODO: explain shape of vectors at each stage. THIS WILL SCREW YOU UP
"""

import base64
from os import listdir
from os.path import isfile, join
import numpy as np
import imageio
import json
from kafka import KafkaProducer
from common import TEST_SAMPLE_COUNT

BASE_DIR="/tmp/images/"
IMG_SHAPE = [1, 240, 320, 1]
DATA_SHAPE = [76800]
# TEST_SAMPLE_COUNT=10


def vectorize_image(img_path):
    """Given an image, represent it as a properly shaped vector"""
    im = imageio.imread(img_path,
                        as_gray=True,
                        pilmode="I").astype(np.float32)

    im = im.reshape(DATA_SHAPE)

    return im

def vectorize_encoded_image(encoded_image):
    """Takes a base64 encoded image and converts it to a tensor"""

    img_bytes = base64.b64decode(encoded_image)
    im = imageio.imread(img_bytes,
                        as_gray=True,
                        pilmode="I").astype(np.float32)

    im = im.reshape(DATA_SHAPE)

    return im

def sample_label(img_path):
    """labels sample based on path"""
    is_game = True if "game" in img_path else False
    if is_game:
        return 1
    else:
        return 0

def vectorize_training_sample(img_path):
    """
    create a single vector from an image

    The first column is the label: 1 for game 0 for not game the rest of the
    data is a reshaped 240x320 representation of a greyscale image
    """
    im = vectorize_image(img_path)
    label = sample_label(img_path)

    return np.hstack(([label], im))

def encode_file(filename):
    """encode file to base64"""

    with open(BASE_DIR + filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    return encoded_string


def all_files_in_dir():
    """Loop through all the files in the directory"""
    # FIXME: make this a generator
    return [f for f in listdir(BASE_DIR) if isfile(join(BASE_DIR, f))]

def encode_all_files(n):
    i = 0
    for file in all_files_in_dir():
        enc = encode_file(file)
        yield enc
        i += 1
        if i and i >= n:
            break

def create_sample_from_file(n=None):
    i = 0
    for file in all_files_in_dir():
        vec = vectorize_training_sample(BASE_DIR + file)
        yield vec
        i += 1
        if i and i >= n:
            break

PRODUCER = KafkaProducer(
    bootstrap_servers="10.1.2.206:9092",
    # NOTE: use this for sample data, not encoded data
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
)

def write_msg_to_queue(msg):

    # NOTE: KafkaProducer takes care of json'ifying the message
    PRODUCER.send('supervised_vectorized_images', msg)

def push_encoded_image_to_queue(msg):

    # NOTE: KafkaProducer takes care of json'ifying the message
    PRODUCER.send('unsupervised_images', msg)

def write_images_to_queue(cnt=None):
    """Writes vectorized images from filesystem to queue"""

    if cnt:
        stop_cnt = cnt
    else:
        stop_cnt = len(all_files_in_dir())

    # FIXME: this takes 8 hours for a single process writing
    # ~ 180000 files
    i = 0
    for vec in create_sample_from_file(stop_cnt):
        write_msg_to_queue({"sample" : vec.tolist()})
        i += 1
        if i % 1000 == 0:
            print("%s samples vectorized" % i)
        # print(".", end='', sep='', flush=True)
    print("Done! Wrote %s records" % i)

def write_encoded_images_to_queue(cnt=None):
    """Writes vectorized images from filesystem to queue"""

    if cnt:
        stop_cnt = cnt
    else:
        stop_cnt = len(all_files_in_dir())

    for encoded_image in encode_all_files(stop_cnt):
        push_encoded_image_to_queue(encoded_image)
        print(".", end='', sep='', flush=True)
    print("done")

if __name__ == "__main__":
    # file = "ad-2017-09-09-174712-1.jpg"
    # file = "game-2017-09-09-174959-1005.jpg"
    # path = BASE_DIR + file
    # sample = vectorize_training_sample(path)

    # jsonified_sample = json.dumps({"sample": sample.tolist()})

    # enc = encode_file(file)
    # print(enc)
    print("vectorizing %s samples" % TEST_SAMPLE_COUNT)
    write_images_to_queue(cnt=TEST_SAMPLE_COUNT)
    # write_encoded_images_to_queue(TEST_SAMPLE_COUNT)
