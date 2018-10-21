"""
Reads image data from Kafka and classifies
"""

from keras.models import load_model
import numpy as np

def fetch_model(filepath="/tmp/bin"):
    return load_model(filepath + "/football_cnn.h5")

def make_prediction(model, X):
    X = X.reshape([1, 240, 320, 1])
    return model.predict(X, batch_size=None, verbose=1, steps=None)

if __name__ == "__main__":
    #NOTE: I'm testing to see if this works:
    basepath = "/tmp/bin"
    samples = np.load(basepath + "/normalized_all_samples.npy")
    labels = np.load(basepath + "/all_labels.npy")

    model = fetch_model()


    for i in range(10):
        pred = make_prediction(model, samples[i])
        print("pred %s, actual %s" % (pred,labels[i]))
