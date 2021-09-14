import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import shutil
from tqdm import tqdm


def split_images(image_dir: str, train_ratio: float = 0.8, extension: str = "png"):
    files = os.listdir(image_dir)
    random.shuffle(files)

    images = [file_name for file_name in files if file_name.endswith(f".{extension}")]
    test_images = images[-10:]
    images = images[:-10]

    train_images = images[: int(len(images) * train_ratio)]
    validation_images = images[int(len(images) * train_ratio) :]

    if os.path.exists(f"{image_dir}/train"):
        shutil.rmtree(f"{image_dir}/train")
    os.makedirs(f"{image_dir}/train")

    if os.path.exists(f"{image_dir}/validate"):
        shutil.rmtree(f"{image_dir}/validate")
    os.makedirs(f"{image_dir}/validate")

    if os.path.exists(f"{image_dir}/test"):
        shutil.rmtree(f"{image_dir}/test")
    os.makedirs(f"{image_dir}/test")

    train_bar = tqdm(train_images)
    for train_image in train_bar:
        train_bar.set_description("Moving training data")
        shutil.move(f"{image_dir}/{train_image}", f"{image_dir}/train/")

    validate_bar = tqdm(validation_images)
    for validation_image in validate_bar:
        validate_bar.set_description("Moving validation data")
        shutil.move(f"{image_dir}/{validation_image}", f"{image_dir}/validate/")

    test_bar = tqdm(test_images)
    for test_image in test_bar:
        test_bar.set_description("Moving validation data")
        shutil.move(f"{image_dir}/{test_image}", f"{image_dir}/test/")


def cropping_images(
    image_dir: str,
    height: int = 128,
    width: int = 128,
    crop_num: int = 20,
    extension: str = "png",
):
    if not os.path.exists(f"{image_dir}/high_resolution"):
        os.makedirs(f"{image_dir}/high_resolution")

    for img_file in tqdm(glob.glob(f"{image_dir}/*.{extension}")):
        image = np.array(Image.open(img_file))
        filename_wo_ext = img_file.rsplit(".", 1)[0]
        pure_filename_wo_ext = filename_wo_ext.rsplit("/", 1)[-1]
        for num in range(crop_num):
            i = np.random.randint(0, image.shape[0] - height - 1)
            j = np.random.randint(0, image.shape[1] - width - 1)

            hr = image[i : i + 32, j : j + 32]
            plt.imsave(
                f"{image_dir}/high_resolution/{pure_filename_wo_ext}_{num}.{extension}",
                hr,
            )


if __name__ == "__main__":
    # split_images("./datasets/DIV2K_2/")
    # cropping_images("./datasets/DIV2K_2/train")
    cropping_images("./datasets/DIV2K_2/validate")
