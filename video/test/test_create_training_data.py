import numpy as np


def test_order_doesnt_matter_for_centering_and_normalizing():

    num_rows = 100
    samples=np.empty([num_rows, 240, 320, 1], dtype=np.float16)

    for row_num in range(num_rows):
        samples[row_num] = np.random.rand(76800).reshape((240, 320, 1))

    mean_then_norm = samples.copy()
    mean_then_norm -= np.mean(mean_then_norm)
    mean_then_norm /= 255

    norm_then_mean = samples.copy()
    norm_then_mean /= 255
    norm_then_mean -= np.mean(norm_then_mean)

    # floating point comparison!
    assert (mean_then_norm - norm_then_mean).mean() < 1e-16
