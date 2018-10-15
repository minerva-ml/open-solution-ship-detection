import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.transform import resize

from common_blocks.utils.misc import get_crop_pad_sequence


def resize_image(image, target_size):
    """Resize image to target size

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Resized image of shape (C x H x W).

    """
    n_channels = image.shape[0]
    resized_image = resize(image, (n_channels, target_size[0], target_size[1]), mode='constant')
    return resized_image


def crop_image(image, target_size):
    """Crop image to target size. Image cropped symmetrically.

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Cropped image of shape (C x H x W).

    """
    top_crop, right_crop, bottom_crop, left_crop = get_crop_pad_sequence(image.shape[1] - target_size[0],
                                                                         image.shape[2] - target_size[1])
    cropped_image = image[:, top_crop:image.shape[1] - bottom_crop, left_crop:image.shape[2] - right_crop]
    return cropped_image


def binarize(image, threshold):
    image_binarized = (image[1, :, :] > threshold).astype(np.uint8)
    return image_binarized


def get_class(prediction, threshold):
    return int(prediction > threshold)


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled


def masks_to_bounding_boxes(labeled_mask):
    if labeled_mask.max() == 0:
        return labeled_mask
    else:
        img_box = np.zeros_like(labeled_mask)
        for label_id in range(1, labeled_mask.max() + 1, 1):
            label = np.where(labeled_mask == label_id, 1, 0).astype(np.uint8)
            _, cnt, _ = cv2.findContours(label, 1, 2)
            rect = cv2.minAreaRect(cnt[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_box, [box], 0, label_id, -1)
        return img_box
