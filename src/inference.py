from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from model import make_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def inference(image):
    high = Image.open(image.replace("low", "high"))

    image = Image.open(image)
    validate_image = np.array(
        image,
        dtype=np.float16,
    )
    validate_image = validate_image / 255.0
    height, width, ch = validate_image.shape

    gen_model = make_generator()
    gen_model.load_weights("./checkpoint/gan_train/generator_best")

    output = (
        (
            gen_model(
                validate_image.reshape([1, height, width, ch]),
                training=False,
            )
            * 127.5
            + 127.5
        )
        .numpy()
        .astype(np.uint8)
    )
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(131)
    ax.imshow(image)
    ax.set_title("Low resolution")
    ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)

    ax = fig.add_subplot(132)
    ax.imshow(high)
    ax.set_title("Original")
    ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)

    ax = fig.add_subplot(133)
    ax.imshow(output[0])
    ax.set_title("Super resolution")
    ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)

    plt.show()


if __name__ == "__main__":
    image = "./datasets/DIV2K_train_HR/validate/low_resolution/00046.png"

    inference(image)
