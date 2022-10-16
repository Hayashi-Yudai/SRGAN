import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET = "DIV2K_train_HR"
HIGH_RESOLUTION_SIZE = 128
NUM_FROM_ONE_IMG = 10


def split_data(path: str) -> list[str]:
    images = os.listdir(path)
    train, test = train_test_split(images, test_size=0.4)
    valid, test = train_test_split(test, test_size=0.5)

    train = [f"{path}/{img}" for img in train]
    valid = [f"{path}/{img}" for img in valid]
    test = [f"{path}/{img}" for img in test]

    return train, valid, test


def make_data(
    path: str,
    dataset_name: str,
    size: tuple[int, int],
    crop_num_per_img: int,
    mode: str,
) -> None:
    """
    Args
    ----
        path [str]: path to the source image
        dataset_name [str]: name of dataset
        size [int]: the size of high-resolution image
        crop_num_per_img [int]: number of images create from an image
        mode [str]: train, validate or test
    """
    img = Image.open(path)
    width, height = img.size

    img_name = path.rsplit("/", 1)[1]
    img_name, ext = img_name.rsplit(".", 1)

    for i in range(crop_num_per_img):
        x = np.random.choice([i for i in range(width - size)], 1)[0]
        y = np.random.choice([i for i in range(height - size)], 1)[0]
        cropped = img.copy().crop((x, y, x + size, y + size))

        noise = np.random.randint(low=-10, high=10, size=(size, size, 3))
        syn_img = np.array(cropped) + noise
        syn_img[syn_img > 255] = 255
        syn_img[syn_img < 0] = 0
        noised_img = Image.fromarray(syn_img.astype(np.uint8))

        cropped_lr = noised_img.resize((size // 4, size // 4))

        cropped.save(
            f"./datasets/{dataset_name}/{mode}/high_resolution/{img_name}_{i}.{ext}"
        )
        cropped_lr.save(
            f"./datasets/{dataset_name}/{mode}/low_resolution/{img_name}_{i}.{ext}"
        )


if __name__ == "__main__":
    images_path = f"./datasets/{DATASET}/original_images"
    train, valid, test = split_data(images_path)
    imgs: dict[str, list[str]] = {"train": train, "validate": valid, "test": test}

    print("Train: ", len(train), "images")
    print("Validation: ", len(valid), "images")
    print("Test: ", len(test), "images")

    print("Train data: ", train[:10], "...")

    for mode in ["train", "validate", "test"]:
        print("mode: ", mode)
        for img in tqdm(imgs[mode]):
            make_data(img, DATASET, HIGH_RESOLUTION_SIZE, NUM_FROM_ONE_IMG, mode)
