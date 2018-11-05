"""
Reads image data from Kafka and classifies
"""

import base64
import imageio
import datetime
from keras.models import load_model
import numpy as np
from create_samples_from_images import vectorize_image
from common import PREDICTION_CONSUMER

# TODO: this center value should not be copied/pasted
CENTER_VALUE=85.30501667317708


def fetch_model(filepath="/tmp/models_and_training_data"):
    return load_model(filepath + "/football_cnn.h5")


def write_image(img_vec):
    fname = datetime.datetime.now().isoformat()
    imageio.imwrite("/tmp/streamed_images/" + fname + ".jpg", img_vec)


import sys
def make_prediction(model, img_vec):
    img_vec = img_vec.reshape([240, 320, 1]).astype(np.float16)
    write_image(img_vec)

    img_vec -= CENTER_VALUE
    img_vec /= 255
    img_vec *= 100

    pred_img = img_vec.reshape([1, 240, 320, 1]).astype(np.int8)

    return model.predict(pred_img,
                         batch_size=None,
                         verbose=1,
                         steps=None)


def print_prediction(prediction):
    prediction = prediction[0]
    game_conf, not_game_conf = prediction

    if game_conf > not_game_conf:
        pred = "GAME!"
    else:
        pred = "NOT GAME!"

    print("prediction: %s, LABEL: %s" % (prediction, pred))


def consume_prediction_queue():
    """docstring for consume_queue"""
    for msg in PREDICTION_CONSUMER:
        yield(msg)


def decode_image(img):
    """Takes an encoded image and decodes it"""
    return base64.b64decode(img)


def main():
    model = fetch_model()

    game_tot, not_game_tot = 0.0, 0.0
    for msg in consume_prediction_queue():
        try:
            encoded_image = msg.value
            decoded_image = decode_image(encoded_image)
            vectored_image = vectorize_image(decoded_image)

            prediction = make_prediction(model, vectored_image)
            game, not_game = prediction[0]
            game_tot += game
            not_game_tot += not_game

            print_prediction(prediction)
            print("game_tot %s, not_game_tot %s" % (game_tot, not_game_tot))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    # #NOTE: I'm testing to see if this works:
    # basepath = "/tmp/bin"
    # samples = np.load(basepath + "/normalized_all_samples.npy")
    # labels = np.load(basepath + "/all_labels.npy")

    # for i in range(10):
    # pred = make_prediction(model, samples[i])
    # print("pred %s, actual %s" % (pred,labels[i]))
    main()

