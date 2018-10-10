import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.externals import joblib

from .masks import get_overlayed_mask


def read_masks_from_csv(image_ids, solution_file_path, image_sizes):
    solution = pd.read_csv(solution_file_path)
    masks = []
    for image_id, image_size in zip(image_ids, image_sizes):
        image_id_pd = image_id + ".jpg"
        mask = get_overlayed_mask(solution.query('ImageId == @image_id_pd'), image_size, labeled=True)
        masks.append(mask)
    return masks


def read_masks(masks_filepaths):
    masks = []
    for mask_filepath in tqdm(masks_filepaths):
        mask = joblib.load(mask_filepath)
        if isinstance(mask, tuple):
            mask = np.zeros(mask).astype(np.uint8)
        masks.append(mask)
    return masks


def read_gt_subset(annotation_file_path, image_ids):
    solution = pd.read_csv(annotation_file_path)
    return solution.query('ImageId in @image_ids')


def read_images(filepaths):
    images = []
    for filepath in filepaths:
        image = np.array(Image.open(filepath))
        images.append(image)
    return images
