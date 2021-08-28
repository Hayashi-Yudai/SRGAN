import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# from tensorflow.keras import mixed_precision

from model import build_model
from losses import content_mse_loss


EPOCHS = 100
BATCH_SIZE = 16
IMG_HEIGHT = 32
IMG_WIDTH = 32
LEARNING_RATE = 1e-4
TRAIN_DATA_PATH = "gs://div2k_dataset/train.tfrecords"
VALIDATE_DATA_PATH = "gs://div2k_dataset/valid.tfrecords"
START_EPOCH = 26
USE_WEIGHT = True


def prepare_from_tfrecords():
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

    return parsed_train_dataset, parsed_valid_dataset


def train_generator(
    gen_model, disc_model, partial_vgg, optimizer, input_high, input_low
):
    with tf.GradientTape() as g_tape:
        g_output = gen_model(input_low)
        d_output = disc_model(g_output)
        g_loss = content_mse_loss(input_high, g_output, d_output, model=partial_vgg)

    g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
    optimizer.apply_gradients(
        grads_and_vars=zip(g_grads, gen_model.trainable_variables)
    )

    return g_loss


def train_discriminator(
    gen_model, disc_model, loss_fn, optimizer, input_high, input_low
):
    with tf.GradientTape() as d_tape:
        d_output_fake = disc_model(gen_model(input_low))
        labels_fake = tf.zeros_like(d_output_fake)

        d_output_real = disc_model(input_high)
        labels_real = tf.ones_like(d_output_real)

        d_loss = loss_fn(y_true=labels_fake, y_pred=d_output_fake)
        d_loss += loss_fn(y_true=labels_real, y_pred=d_output_real)

    d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
    optimizer.apply_gradients(
        grads_and_vars=zip(d_grads, disc_model.trainable_variables)
    )

    return d_loss


def validate_generator(gen_model, disc_model, partial_vgg, input_low, input_high):
    g_output = gen_model(input_low, training=False)
    d_output = disc_model(g_output, training=False)
    g_loss = content_mse_loss(input_high, g_output, d_output, model=partial_vgg)

    return g_loss


def train(device_name):
    with tf.device(device_name):
        gen_model, disc_model, model = build_model(IMG_HEIGHT, IMG_WIDTH, USE_WEIGHT)

    train_data, valid_data = prepare_from_tfrecords()
    disc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "./logs/" + current_time + "/train"
    valid_log_dir = "./logs/" + current_time + "/valid"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    smallest_g_loss = 1e9
    if USE_WEIGHT:
        smallest_g_loss = 0.05602

    for epoch in range(START_EPOCH, EPOCHS):
        g_losses = []
        d_losses = []

        for images in tqdm(train_data):
            g_loss = train_generator(
                gen_model=gen_model,
                disc_model=disc_model,
                partial_vgg=model,
                optimizer=gen_optimizer,
                input_high=images["high"],
                input_low=images["low"],
            )
            d_loss = train_discriminator(
                gen_model=gen_model,
                disc_model=disc_model,
                loss_fn=disc_loss,
                optimizer=disc_optimizer,
                input_high=images["high"],
                input_low=images["low"],
            )

            g_losses.append(g_loss.numpy())
            d_losses.append(d_loss.numpy())

        g_loss_mean = np.mean(g_losses)
        d_loss_mean = np.mean(d_losses)
        print(
            f"Epoch {epoch + 1}| Generator-Loss: {g_loss_mean:.3e},",
            f"Discriminator-Loss: {d_loss_mean:.3e}",
        )

        g_valid_losses = []
        for images in tqdm(valid_data):
            gen_loss = validate_generator(
                gen_model, disc_model, model, images["low"], images["high"]
            )
            g_valid_losses.append(gen_loss)
        valid_loss = np.mean(g_valid_losses)
        print(f"Validation| Generator-Loss: {valid_loss:.3e}")

        if valid_loss < smallest_g_loss:
            gen_model.save_weights("./checkpoint/test/generator_best")
            disc_model.save_weights("./checkpoint/test/discriminator_best")

            smallest_g_loss = valid_loss
            print("Model saved")

        with train_summary_writer.as_default():
            tf.summary.scalar("g_loss", g_loss_mean, step=epoch)
            tf.summary.scalar("d_loss", d_loss_mean, step=epoch)
        with valid_summary_writer.as_default():
            tf.summary.scalar("g_loss", valid_loss, step=epoch)

        gen_model.save_weights("./checkpoint/test/generator_last")
        disc_model.save_weights("./checkpoint/test/discriminator_last")


if __name__ == "__main__":
    if tf.config.list_physical_devices("GPU"):
        device_name = tf.test.gpu_device_name()
        # mixed_precision.set_global_policy("mixed_float16")
    else:
        device_name = "/CPU:0"

    train(device_name)
