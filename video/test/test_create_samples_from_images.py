import base64
import numpy as np
import imageio
import create_samples_from_images as csfi

TEST_FILE = "test/sample_images/gbp_nep_ad_2018-11-04T20:50:01-05:00_0001.jpg"


def parse_image(io_thing, typespec=np.uint8):
    """Helper function to parse image"""

    return imageio.imread(io_thing,
                          as_gray=True,
                          pilmode="I").astype(typespec)

def test_type_equiv():
    img_f32 = parse_image(TEST_FILE, np.float32)
    img_u8 = parse_image(TEST_FILE, np.uint8)

    assert (img_f32 == img_u8).all()

def test_b64_equiv():

    # NOTE: training data is files and prediction data is b64 encoded over the network

    with open(TEST_FILE, 'rb') as f:
        f_contents = f.read()

    b64bits = base64.b64encode(f_contents)
    img_64 = parse_image(base64.b64decode(b64bits))
    img_u8 = parse_image(TEST_FILE)

    assert (img_64 == img_u8).all()
# with open(TEST_FILE, 'rb') as f:
#     f_contents = f.read()
#
# b64 = base64.b16decode(f_contents)
