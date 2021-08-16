import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    PReLU,
    BatchNormalization,
    LeakyReLU,
    Flatten,
)
from tensorflow.keras import Sequential, Model


class BResidualBlock(Model):
    def __init__(self):
        super(BResidualBlock, self).__init__()

        self.conv = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.bn = BatchNormalization()
        self.prelu = PReLU()

        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.bn2 = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.prelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += input_tensor

        return x


class ResidualBlock(Model):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.residual1 = BResidualBlock()
        self.residual2 = BResidualBlock()
        self.residual3 = BResidualBlock()
        self.residual4 = BResidualBlock()
        self.residual5 = BResidualBlock()

        self.conv = Conv2D(filters=64, kernel_size=3, padding="same")
        self.bn = BatchNormalization()

    def call(self, input_tensor):
        x = self.residual1(input_tensor)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)

        x = self.conv(x)
        x = self.bn(x)

        x += input_tensor

        return x


class DescriminatorBlock(Model):
    def __init__(self, filters=128):
        super(DescriminatorBlock, self).__init__()
        self.filters = filters

        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")
        self.bn1 = BatchNormalization()
        self.lrelu1 = LeakyReLU(alpha=0.2)
        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=2, padding="same")
        self.bn2 = BatchNormalization()
        self.lrelu2 = LeakyReLU(alpha=0.2)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)

        return x


class PixelShuffler(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelShuffler, self).__init__()

    def call(self, input_tensor):
        x = tf.nn.depth_to_space(input_tensor, 2)

        return x


def make_generator():
    model = Sequential(
        [
            Conv2D(filters=64, kernel_size=9, padding="same"),
            PReLU(),
            ResidualBlock(),
            Conv2D(filters=256, kernel_size=3, padding="same"),
            PixelShuffler(),
            PReLU(),
            Conv2D(filters=256, kernel_size=3, padding="same"),
            PixelShuffler(),
            PReLU(),
            Conv2D(filters=3, kernel_size=9),
        ]
    )

    return model


def make_descriminator():
    model = Sequential(
        [
            Conv2D(filters=64, kernel_size=3, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2D(filters=64, kernel_size=3, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            DescriminatorBlock(128),
            DescriminatorBlock(256),
            DescriminatorBlock(512),
            Flatten(),
            Dense(1024),
            LeakyReLU(alpha=0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    return model


if __name__ == "__main__":
    model = make_descriminator()
    model.build(input_shape=(None, 256, 256, 3))
    model.summary()
