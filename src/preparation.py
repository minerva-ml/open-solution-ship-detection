import os
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pandas as pd

from .utils import rle_from_binary, get_overlayed_mask
from .pipeline_config import ORIGINAL_SIZE, EXCLUDED_FILENAMES

def overlay_masks(annotation_file_name, target_dir, dev_mode):
    os.makedirs(target_dir, exist_ok=True)
    annotations = pd.read_csv(annotation_file_name)
    if dev_mode:
        annotations = annotations.sample(1000)
    for file_name, image_annotation in annotations.groupby("ImageId"):
        if file_name not in EXCLUDED_FILENAMES:
            target_file_name = os.path.join(target_dir, file_name.split('.')[0])
            mask = get_overlayed_mask(image_annotation, ORIGINAL_SIZE)
            if mask.sum()==0:
                mask = ORIGINAL_SIZE
            save_target_mask(target_file_name, mask)


def prepare_class_encoding(binary_mask):
    segmentation = rle_from_binary(binary_mask.astype(np.uint8))
    segmentation['counts'] = segmentation['counts'].decode("UTF-8")
    return segmentation


def save_target_mask(target_filepath, mask):
    joblib.dump(mask, target_filepath)


def train_valid_split(meta, evaluation_size, validation_size, random_state=None):
    meta_train = meta[meta['is_train'] == 1]

    meta_train_split, meta_valid_split = train_test_split(meta_train,
                                                          test_size=evaluation_size,
                                                          random_state=random_state)

    if validation_size == 1:
        meta_valid = meta_valid_split
    else:
        _, meta_valid = train_test_split(meta_valid_split, test_size=validation_size, random_state=random_state)

    return meta_train_split, meta_valid


def run_length_decoding(mask_rle, shape=(768, 768)):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T