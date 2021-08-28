# SRGAN

## Verification environment

- Python 3.9.6
- Tensorflow 2.6.0

## Network structure

![https://arxiv.org/pdf/1609.04802.pdf](https://github.com/Hayashi-Yudai/SRGAN/blob/main/assets/network_img.png)


## How to train

### Prepare your dataset

Make the following directory tree for your dataset on the project root and place original images in `train/high_resolution` and `validate/high_resolution/` directories.

```
.datasets
└── (your dataset name)
    ├── test
    │   ├── high_resolution
    │   └── low_resolution
    ├── train
    │   ├── high_resolution
    │   └── low_resolution
    └── validate
        ├── high_resolution
        └── low_resolution
```

Next, make low resolution images which have quarter size of original ones and place them in `low_resolution` directories.

This program request TFRecords as dataset. I prepare a function for you. Fix `dataset_name` and `extension` in the src/datasets.py and execute it from project root.

```bash
python src/dataset.py
```

Make sure there exists `train.tfrecords` and `valid.tfrecords` in the `datasets/(your dataset name)` directory.


### Configure parameters

The parameters like hyper-parameters are set in the config.yaml


### Training

```bash
python src/train.py
```
