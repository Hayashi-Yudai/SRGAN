import tensorflow as tf
import yaml


def prepare_from_tfrecords():
    print("Loading dataset ...")
    with open("config.yaml") as yfile:
        config = yaml.safe_load(yfile)

        TRAIN_DATA_PATH = config["TRAIN_DATA_PATH"]
        VALIDATE_DATA_PATH = config["VALIDATE_DATA_PATH"]
        IMG_HEIGHT = config["IMG_HEIGHT"]
        IMG_WIDTH = config["IMG_WIDTH"]
        BATCH_SIZE = config["BATCH_SIZE"]

    # Load training dataset
    raw_image_dataset = tf.data.TFRecordDataset(TRAIN_DATA_PATH)
    image_feature_description = {
        "high_image_raw": tf.io.FixedLenFeature([], tf.string),
        "low_image_raw": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_dataset(example_proto):
        example = tf.io.parse_single_example(example_proto, image_feature_description)
        high_image = tf.reshape(
            tf.io.decode_raw(example["high_image_raw"], tf.uint8),
            (IMG_HEIGHT * 4, IMG_WIDTH * 4, 3),
        )
        low_image = tf.reshape(
            tf.io.decode_raw(example["low_image_raw"], tf.uint8),
            (IMG_HEIGHT, IMG_WIDTH, 3),
        )
        high_image = tf.cast(high_image, tf.float16)
        low_image = tf.cast(low_image, tf.float16)

        high_image = (high_image - 122.5) / 255.0
        low_image = (low_image - 122.5) / 255.0

        return {"high": high_image, "low": low_image}

    parsed_train_dataset = raw_image_dataset.map(_parse_image_dataset).batch(BATCH_SIZE)

    # Load validation dataset
    raw_image_dataset = tf.data.TFRecordDataset(VALIDATE_DATA_PATH)
    image_feature_description = {
        "high_image_raw": tf.io.FixedLenFeature([], tf.string),
        "low_image_raw": tf.io.FixedLenFeature([], tf.string),
    }

    parsed_valid_dataset = raw_image_dataset.map(_parse_image_dataset).batch(BATCH_SIZE)
    print("Dataset is loaded!")

    return parsed_train_dataset, parsed_valid_dataset
