import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image


def prepare_from_tfrecords(train_data, validate_data, height, width, batch_size):
    print("Loading dataset ...")

    # Load training dataset
    raw_image_dataset = tf.data.TFRecordDataset(train_data)
    image_feature_description = {
        "high_image_raw": tf.io.FixedLenFeature([], tf.string),
        "low_image_raw": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_dataset(example_proto):
        example = tf.io.parse_single_example(example_proto, image_feature_description)
        high_image = tf.reshape(
            tf.io.decode_raw(example["high_image_raw"], tf.uint8),
            (height * 4, width * 4, 3),
        )
        low_image = tf.reshape(
            tf.io.decode_raw(example["low_image_raw"], tf.uint8),
            (height, width, 3),
        )
        high_image = tf.cast(high_image, tf.float16)
        low_image = tf.cast(low_image, tf.float16)

        high_image = (high_image - 127.5) / 127.5
        low_image = low_image / 255.0

        return {"high": high_image, "low": low_image}

    parsed_train_dataset = raw_image_dataset.map(_parse_image_dataset).batch(batch_size)
    parsed_train_dataset = parsed_train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )

    # Load validation dataset
    raw_image_dataset = tf.data.TFRecordDataset(validate_data)
    image_feature_description = {
        "high_image_raw": tf.io.FixedLenFeature([], tf.string),
        "low_image_raw": tf.io.FixedLenFeature([], tf.string),
    }

    parsed_valid_dataset = raw_image_dataset.map(_parse_image_dataset).batch(batch_size)
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
