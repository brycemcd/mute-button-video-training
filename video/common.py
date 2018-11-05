"""
Shared values, functions and constants live here
"""
import json
from kafka import KafkaConsumer


# TEST_SAMPLE_COUNT=150000 # NOTE: this is the max my machine's memory can tollerate
TEST_SAMPLE_COUNT=4000
IMG_SHAPE = [TEST_SAMPLE_COUNT, 240, 320, 1]
DATA_SHAPE = [76800]
NOT_GAME_VECTOR = [0, 1]
GAME_VECTOR = [1, 0]


VECTORED_IMAGE_QUEUE = KafkaConsumer('supervised_vectorized_images',
                                     bootstrap_servers='10.1.2.206:9092',
                                     value_deserializer=lambda v: json.loads(v),
                                     # group_id='my_favorite_group1',
                                     auto_offset_reset='earliest',
                                     )

PREDICTION_CONSUMER = KafkaConsumer('unsupervised_images',
                                    bootstrap_servers='10.1.2.206:9092',
                                    # group_id='my_favorite_group1',
                                    auto_offset_reset='earliest',
                                    )
