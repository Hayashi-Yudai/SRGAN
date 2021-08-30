import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
from PIL import Image


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
    parsed_train_dataset = parsed_train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )

    # Load validation dataset
    raw_image_dataset = tf.data.TFRecordDataset(VALIDATE_DATA_PATH)
    image_feature_description = {
        "high_image_raw": tf.io.FixedLenFeature([], tf.string),
        "low_image_raw": tf.io.FixedLenFeature([], tf.string),
    }

    parsed_valid_dataset = raw_image_dataset.map(_parse_image_dataset).batch(BATCH_SIZE)
    parsed_valid_dataset = parsed_valid_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    print("Dataset is loaded!")

    return parsed_train_dataset, parsed_valid_dataset


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tfrecords(dataset_name, extension="png"):
    high_image_files = tf.io.gfile.glob(
        f"./datasets/{dataset_name}/train/high_resolution/*.{extension}"
    )
    low_image_files = [image.replace("high", "low") for image in high_image_files]

    with tf.io.TFRecordWriter(f"./datasets/{dataset_name}/train.tfrecords") as writer:
        for high_img_file, low_img_file in tqdm(zip(high_image_files, low_image_files)):
            high_image_string = (
                np.array(Image.open(high_img_file)).astype(np.uint8).tobytes()
            )
            low_image_string = (
                np.array(Image.open(low_img_file)).astype(np.uint8).tobytes()
            )
            feature = {
                "high_image_raw": _bytes_feature(high_image_string),
                "low_image_raw": _bytes_feature(low_image_string),
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())

    high_image_files = tf.io.gfile.glob(
        f"./datasets/{dataset_name}/validate/high_resolution/*.{extension}"
    )
    low_image_files = [image.replace("high", "low") for image in high_image_files]

    with tf.io.TFRecordWriter(f"./datasets/{dataset_name}/valid.tfrecords") as writer:
        for high_img_file, low_img_file in tqdm(zip(high_image_files, low_image_files)):
            high_image_string = (
                np.array(Image.open(high_img_file)).astype(np.uint8).tobytes()
            )
            low_image_string = (
                np.array(Image.open(low_img_file)).astype(np.uint8).tobytes()
            )
            feature = {
                "high_image_raw": _bytes_feature(high_image_string),
                "low_image_raw": _bytes_feature(low_image_string),
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    dataset_name = "YOUR DATASET NAME"
    extension = "Image extension (png, jpeg, ...)"

    make_tfrecords(dataset_name, extension)
