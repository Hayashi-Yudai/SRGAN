import os
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


if __name__ == "__main__":
    split_images("./datasets/DIV2K_2/")
