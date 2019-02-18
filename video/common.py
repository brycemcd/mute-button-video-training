"""
Shared values, functions and constants live here
"""
import json
from kafka import KafkaConsumer, KafkaProducer
import numpy as np


# TEST_SAMPLE_COUNT=150000 # NOTE: this is the max my machine's memory can tollerate
TEST_SAMPLE_COUNT = 22000
IMG_SHAPE = [TEST_SAMPLE_COUNT, 240, 320, 1]
DATA_SHAPE = [76800]

GAME_LABEL = 1
NOT_GAME_LABEL = 0

GAME_VECTOR = [GAME_LABEL, NOT_GAME_LABEL]
NOT_GAME_VECTOR = [NOT_GAME_LABEL, GAME_LABEL]


KAFKA_BOOTSTRAP_SERVERS = ["10.1.2.230:9092", "10.1.2.244:9092", "10.1.5.207:9092"]

PRODUCER = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    # NOTE: use this for sample data, not encoded data
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
)

MODEL_PRODUCER = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    # value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    # NOTE: use this for sample data, not encoded data
)

# VECTORED_IMAGE_QUEUE = KafkaConsumer('supervised_vectorized_images',
                                     # bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                                     # value_deserializer=lambda v: json.loads(v),
                                     # # group_id='my_favorite_group1',
                                     # auto_offset_reset='earliest',
                                     # )

QUEUE_MODEL_NAME="footballModelsArchitectures"
MODEL_CONSUMER = KafkaConsumer(QUEUE_MODEL_NAME,
                               bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                               value_deserializer=lambda v: json.loads(v),
                               group_id='model_consumers',
                               auto_offset_reset='latest',
)
# PREDICTION_CONSUMER = KafkaConsumer('unsupervised_images',
                                    # bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                                    # # group_id='my_favorite_group1',
                                    # auto_offset_reset='latest',
                                    # )

def center_sample(sample, center_value):
    """Provides consistent interface to change tensor values for
    training and prediction
    """
    sample = sample.astype(np.float16)

    sample -= center_value
    sample /= 255
    sample *= 100

    # NOTE: casting this back to an int has the effect of rounding
    # We lose precision but gain memory space (8 vs. 16 or 32 bits per value)
    sample = sample.astype(np.int8)

    return sample
