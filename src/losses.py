import tensorflow as tf


def adversarial_loss(y_pred: tf.Tensor) -> tf.Tensor:
    # Add penalty only for fake input
    return -tf.reduce_mean(tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))


def content_mse_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, d_output, model: tf.keras.Model
) -> tf.Tensor:
    adv_loss = adversarial_loss(d_output)
    # mse_loss = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

    y_pred = model(y_pred)
    y_true = model(y_true)
    content_loss = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

    return content_loss + adv_loss * 1e-3
