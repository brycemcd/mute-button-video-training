"""
Given a directory of images, this script will base64 encode the image and write
it to a Kafka queue


For testing, I did this:
 ffmpeg -f image2 -pattern_type glob -i "./gbp_nep_game_*_001*.jpg" -vf "scale=320:240, format=gray" converted_images/2018-11-05-game-%d.jpg
 ffmpeg -f image2 -pattern_type glob -i "./gbp_nep_game_*_001*.jpg" -vf "scale=320:240, format=gray" converted_images/2018-11-05-game-%d.jpg
 ffmpeg -f image2 -pattern_type glob -i "./*game_*.jpg" -vf "scale=320:240, format=gray" converted_images/2018-11-05-game-%d.jpg
"""

import base64
from os import listdir
from os.path import isfile, join
import numpy as np
import imageio
from common import TEST_SAMPLE_COUNT, PRODUCER, IMG_SHAPE, DATA_SHAPE, GAME_LABEL, NOT_GAME_LABEL

BASE_DIR="/tmp/test/converted_images/"


def vectorize_image(img_path):
    """Given an image, represent it as a properly shaped vector"""

    im = imageio.imread(img_path,
                        as_gray=True,
                        pilmode="I").astype(np.uint8)

    im = im.reshape(DATA_SHAPE)

    return im


def vectorize_encoded_image(encoded_image):
    """Takes a base64 encoded image and converts it to a tensor"""

    img_bytes = base64.b64decode(encoded_image)
    im = vectorize_image(img_bytes)

    return im


def sample_label(img_path):
    """labels sample based on path"""
    is_game = True if "game" in img_path else False
    if is_game:
        return GAME_LABEL
    else:
        return NOT_GAME_LABEL


def vectorize_training_sample(img_path):
    """
    create a single vector from an image

    The first column is the label: 1 for game 0 for not game the rest of the
    data is a reshaped 240x320 representation of a greyscale image
    """
    im = vectorize_image(img_path)
    label = sample_label(img_path)

    return np.hstack(([label], im))


def all_files_in_dir():
    """Loop through all the files in the directory"""

    # FIXME: make this a generator
    return [f for f in listdir(BASE_DIR) if isfile(join(BASE_DIR, f))]


def write_msg_to_queue(msg):

    # NOTE: KafkaProducer takes care of json'ifying the message
    PRODUCER.send('supervised_vectorized_images', msg)


def main():
    """Run the expected routine"""
    i = 0
    for file in all_files_in_dir():
        vec = vectorize_training_sample(BASE_DIR + file)
        write_msg_to_queue({"sample" : vec.tolist()})
        print(".", end='', sep='', flush=True)
        i += 1
        if i % 1000 == 0:
            print("%s written" % i)


if __name__ == "__main__":
    print("vectorizing %s samples" % TEST_SAMPLE_COUNT)
    main()
    print("done!")
