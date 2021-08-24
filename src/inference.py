from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import make_generator


def inference(image):
    validate_image = np.array(
        Image.open(image),
        dtype=np.float16,
    )
    validate_image = (validate_image - 122.5) / 255.0
    height, width, ch = validate_image.shape

    gen_model = make_generator()
    gen_model.load_weights("./checkpoint/generator")

    output = (
        (
            gen_model(
                validate_image.reshape([1, height, width, ch]),
                training=False,
            )
            * 255
            + 122.5
        )
        .numpy()
        .astype(np.uint8)
    )
    plt.figure()
    plt.imshow(output[0])
    plt.show()
