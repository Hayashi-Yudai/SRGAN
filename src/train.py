import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml

# from tensorflow.keras import mixed_precision

from model import build_model
from losses import content_mse_loss
from dataset import prepare_from_tfrecords


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


class SummaryWriter:
    def __init__(self):
        self.epoch = START_EPOCH
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "./logs/" + current_time + "/train"
        valid_log_dir = "./logs/" + current_time + "/valid"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    def write(self, g_loss_mean, d_loss_mean, valid_loss):
        with self.train_summary_writer.as_default():
            tf.summary.scalar("g_loss", g_loss_mean, step=self.epoch)
            tf.summary.scalar("d_loss", d_loss_mean, step=self.epoch)
        with self.valid_summary_writer.as_default():
            tf.summary.scalar("g_loss", valid_loss, step=self.epoch)

        self.epoch += 1


def train(device_name):
    with tf.device(device_name):
        gen_model, disc_model, model = build_model(IMG_HEIGHT, IMG_WIDTH, WEIGHT)

    train_data, valid_data = prepare_from_tfrecords()
    disc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    writer = SummaryWriter()

    smallest_g_loss = 1e9
    if WEIGHT != "":
        smallest_g_loss = G_LOSS

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
            gen_model.save_weights(f"{CHECKPOINT_PATH}/generator_best")
            disc_model.save_weights(f"{CHECKPOINT_PATH}/discriminator_best")

            smallest_g_loss = valid_loss
            print("Model saved")

        writer.write(g_loss_mean, d_loss_mean, valid_loss)

        gen_model.save_weights(f"{CHECKPOINT_PATH}/generator_last")
        disc_model.save_weights(f"{CHECKPOINT_PATH}/discriminator_last")


if __name__ == "__main__":
    if tf.config.list_physical_devices("GPU"):
        device_name = tf.test.gpu_device_name()
        # mixed_precision.set_global_policy("mixed_float16")
    else:
        device_name = "/CPU:0"

    with open("config.yaml") as yfile:
        config = yaml.safe_load(yfile)

        EPOCHS = config["EPOCHS"]
        IMG_HEIGHT = config["IMG_HEIGHT"]
        IMG_WIDTH = config["IMG_WIDTH"]
        LEARNING_RATE = config["LEARNING_RATE"]
        CHECKPOINT_PATH = config["CHECKPOINT_PATH"]
        START_EPOCH = config["START_EPOCH"]
        WEIGHT = config["WEIGHT"]
        G_LOSS = config["G_LOSS"]

    train(device_name)
