import datetime
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import yaml

from model import make_generator, make_discriminator, make_vgg
from dataset import prepare_from_tfrecords


class SRResNetTrainer:
    def __init__(
        self,
        epochs: int = 10000,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        training_data_path: str = "./datasets/train.tfrecords",
        validate_data_path: str = "./datasets/valid.tfrecords",
        height: int = 32,
        width: int = 32,
        g_weight: str = None,
        checkpoint_path: str = "./checkpoint",
        best_generator_loss: float = 1e9,
    ):
        self.epochs = epochs
        self.batch_size = batch_size

        self.generator = make_generator()
        if g_weight is not None and g_weight != "":
            print("Loading weights on generator...")
            self.generator.load_weights(g_weight)

        self.train_data, self.validate_data = prepare_from_tfrecords(
            train_data=training_data_path,
            validate_data=validate_data_path,
            height=height,
            width=width,
            batch_size=batch_size,
        )
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.best_generator_loss = best_generator_loss
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.checkpoint_path = checkpoint_path
        self.make_checkpoint = len(checkpoint_path) > 0

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "./logs/" + current_time + "/train_generator"
        valid_log_dir = "./logs/" + current_time + "/valid_generator"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    @tf.function
    def train_step(self, lr: tf.Tensor, hr: tf.Tensor):
        with tf.GradientTape() as tape:
            generated_fake = self.generator(lr)
            g_loss = self.mse_loss(generated_fake, hr)

        generator_grad = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            grads_and_vars=zip(generator_grad, self.generator.trainable_variables)
        )

        return g_loss

    @tf.function
    def validation_step(self, lr: tf.Tensor, hr: tf.Tensor):
        generated_fake = self.generator(lr)
        g_loss = self.mse_loss(generated_fake, hr)

        return g_loss

    def train(self, start_epoch=0):
        for step in range(start_epoch, self.epochs):
            g_loss_train = []
            for images in tqdm(self.train_data):
                g_loss = self.train_step(images["low"], images["high"])
                g_loss_train.append(g_loss.numpy())

            g_loss_train_mean = np.mean(g_loss_train)

            with self.train_summary_writer.as_default():
                tf.summary.scalar("g_loss", g_loss_train_mean, step=step)

            print(
                f"Epoch {step+ 1}| Generator-Loss: {g_loss_train_mean:.3e},",
            )

            g_loss_valid = []
            for images in tqdm(self.validate_data):
                g_loss = self.validation_step(images["low"], images["high"])
                g_loss_valid.append(g_loss)

            g_loss_valid_mean = np.mean(g_loss_valid)

            with self.valid_summary_writer.as_default():
                tf.summary.scalar("g_loss", g_loss_valid_mean, step=step)

            print(
                f"Validation| Generator-Loss: {g_loss_valid_mean:.3e},",
            )

            if self.make_checkpoint:
                self.generator.save_weights(f"{self.checkpoint_path}/generator_last")

                if g_loss_valid_mean < self.best_generator_loss:
                    self.best_generator_loss = g_loss_valid_mean
                    self.generator.save_weights(
                        f"{self.checkpoint_path}/generator_best"
                    )

                    print("Model Saved")


class SRGANTrainer:
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        height: int = 32,
        width: int = 32,
        g_weight: str = None,
        d_weight: str = None,
        training_data_path: str = "./datasets/train.tfrecords",
        validate_data_path: str = "./datasets/valid.tfrecords",
        checkpoint_path: str = "./checkpoints",
        best_generator_loss: float = 1e9,
    ):
        # -----------------------------
        # Hyper-parameters
        # -----------------------------
        self.epochs = epochs
        self.batch_size = batch_size

        # -----------------------------
        # Model
        # -----------------------------
        self.generator = make_generator()
        self.discriminator = make_discriminator()
        self.vgg = make_vgg(height=height, width=width)

        if g_weight is not None and g_weight != "":
            print("Loading weights on generator...")
            self.generator.load_weights(g_weight)
        if d_weight is not None and d_weight != "":
            print("Loading weights on discriminator...")
            self.discriminator.load_weights(d_weight)

        # -----------------------------
        # Data
        # -----------------------------
        self.train_data, self.validate_data = prepare_from_tfrecords(
            train_data=training_data_path,
            validate_data=validate_data_path,
            height=height,
            width=width,
            batch_size=batch_size,
        )

        # -----------------------------
        # Loss
        # -----------------------------
        self.discriminator_loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=False
        )
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.best_generator_loss = best_generator_loss

        # -----------------------------
        # Optimizer
        # -----------------------------
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate)

        # -----------------------------
        # Summary Writer
        # -----------------------------
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "./logs/" + current_time + "/train"
        valid_log_dir = "./logs/" + current_time + "/valid"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        self.checkpoint_path = checkpoint_path
        self.make_checkpoint = len(checkpoint_path) > 0

    @tf.function
    def _content_loss(self, lr: tf.Tensor, hr: tf.Tensor):
        lr_vgg = self.vgg(lr) / 12.75
        hr_vgg = self.vgg(hr) / 12.75

        return self.mse_loss(lr_vgg, hr_vgg)

    def _adversarial_loss(self, output):
        return self.bce_loss(tf.ones_like(output), output)

    @tf.function
    def train_step(self, lr: tf.Tensor, hr: tf.Tensor) -> tuple[tf.Tensor]:
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated_fake = self.generator(lr)

            real = self.discriminator(hr)
            fake = self.discriminator(generated_fake)

            d_loss = self.discriminator_loss_fn(real, tf.ones_like(real))
            d_loss += self.discriminator_loss_fn(fake, tf.zeros_like(fake))

            g_loss = self._content_loss(generated_fake, hr)
            g_loss += self._adversarial_loss(generated_fake) * 1e-3

        discriminator_grad = d_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        generator_grad = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            grads_and_vars=zip(
                discriminator_grad, self.discriminator.trainable_variables
            )
        )
        self.generator_optimizer.apply_gradients(
            grads_and_vars=zip(generator_grad, self.generator.trainable_variables)
        )

        return g_loss, d_loss

    @tf.function
    def validation_step(self, lr: tf.Tensor, hr: tf.Tensor):
        generated_fake = self.generator(lr)
        real = self.discriminator(hr)
        fake = self.discriminator(generated_fake)

        d_loss = self.discriminator_loss_fn(real, tf.ones_like(real))
        d_loss += self.discriminator_loss_fn(fake, tf.zeros_like(fake))

        g_loss = self._content_loss(generated_fake, hr)
        g_loss += self._adversarial_loss(generated_fake) * 1e-3

        return g_loss, d_loss

    def train(self, start_epoch):
        for step in range(start_epoch, self.epochs):
            d_loss_train = []
            g_loss_train = []
            for images in tqdm(self.train_data):
                g_loss, d_loss = self.train_step(images["low"], images["high"])
                g_loss_train.append(g_loss.numpy())
                d_loss_train.append(d_loss.numpy())

            g_loss_train_mean = np.mean(g_loss_train)
            d_loss_train_mean = np.mean(d_loss_train)

            with self.train_summary_writer.as_default():
                tf.summary.scalar("g_loss", g_loss_train_mean, step=step)
                tf.summary.scalar("d_loss", d_loss_train_mean, step=step)

            print(
                f"Epoch {step+ 1}| Generator-Loss: {g_loss_train_mean:.3e},",
                f"Discriminator-Loss: {d_loss_train_mean:.3e}",
            )

            d_loss_valid = []
            g_loss_valid = []
            for images in tqdm(self.validate_data):
                g_loss, d_loss = self.validation_step(images["low"], images["high"])
                d_loss_valid.append(d_loss)
                g_loss_valid.append(g_loss)

            g_loss_valid_mean = np.mean(g_loss_valid)
            d_loss_valid_mean = np.mean(d_loss_valid)

            with self.valid_summary_writer.as_default():
                tf.summary.scalar("g_loss", g_loss_valid_mean, step=step)
                tf.summary.scalar("d_loss", d_loss_valid_mean, step=step)

            print(
                f"Validation| Generator-Loss: {g_loss_valid_mean:.3e},",
                f"Discriminator-Loss: {d_loss_valid_mean:.3e}",
            )

            if self.make_checkpoint:
                self.generator.save_weights(f"{self.checkpoint_path}/generator_last")
                self.discriminator.save_weights(
                    f"{self.checkpoint_path}/discriminator_last"
                )

                if g_loss_valid_mean < self.best_generator_loss:
                    self.best_generator_loss = g_loss_valid_mean
                    self.generator.save_weights(
                        f"{self.checkpoint_path}/generator_best"
                    )
                    self.discriminator.save_weights(
                        f"{self.checkpoint_path}/discriminator_best"
                    )

                    print("Model Saved")


if __name__ == "__main__":
    with open("config.yaml") as yfile:
        config = yaml.safe_load(yfile)

        training_data_path = config["TRAIN_DATA_PATH"]
        validate_data_path = config["VALIDATE_DATA_PATH"]
        height = config["IMG_HEIGHT"]
        width = config["IMG_WIDTH"]
        epochs = config["EPOCHS"]
        batch_size = config["BATCH_SIZE"]
        learning_rate = config["LEARNING_RATE"]
        g_weight = config["GEN_WEIGHT"]
        d_weight = config["DISC_WEIGHT"]
        checkpoint_path = config["CHECKPOINT_PATH"]
        best_generator_loss = config["G_LOSS"]
        start_epoch = config["START_EPOCH"]

    """
    trainer = SRGANTrainer(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_data_path=training_data_path,
        validate_data_path=validate_data_path,
        height=height,
        width=width,
        g_weight=g_weight,
        d_weight=d_weight,
        checkpoint_path=checkpoint_path,
        best_generator_loss=best_generator_loss,
    )
    """
    trainer = SRResNetTrainer(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_data_path=training_data_path,
        validate_data_path=validate_data_path,
        height=height,
        width=width,
        g_weight=g_weight,
        checkpoint_path=checkpoint_path,
        best_generator_loss=best_generator_loss,
    )
    trainer.train(start_epoch=start_epoch)
