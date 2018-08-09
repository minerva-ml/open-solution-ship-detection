from functools import partial

import numpy as np
from tqdm import tqdm
from pycocotools import mask as cocomask

from .utils import get_segmentations, get_overlayed_mask
from .pipeline_config import ORIGINAL_SIZE


def iou(gt, pred):
    gt[gt > 0] = 1.
    pred[pred > 0] = 1.
    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = 1e-09
    return intersection / union


def compute_ious(gt, predictions):
    gt_ = get_segmentations(gt)
    predictions_ = get_segmentations(predictions)

    if len(gt_) == 0 and len(predictions_) == 0:
        return np.ones((1, 1))
    elif len(gt_) != 0 and len(predictions_) == 0:
        return np.zeros((1, 1))
    else:
        iscrowd = [0 for _ in predictions_]
        ious = cocomask.iou(gt_, predictions_, iscrowd)
        if not np.array(ious).size:
            ious = np.zeros((1, 1))
        return ious


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def compute_eval_metric_per_image(gt, predictions, metric_to_average='precision'):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    if metric_to_average == 'precision':
        metric_function = compute_precision_at
    elif metric_to_average[0] == 'f':
        beta = float(metric_to_average[1:])
        metric_function = partial(compute_f_beta_at, beta = beta)
    else:
        raise NotImplementedError
    metric_per_image = [metric_function(ious, th) for th in thresholds]
    return sum(metric_per_image) / len(metric_per_image)


def intersection_over_union(y_true, y_pred):
    ious = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        iou = compute_ious(y_t, y_p)
        iou_mean = 1.0 * np.sum(iou) / len(iou)
        ious.append(iou_mean)
    return np.mean(ious)


def intersection_over_union_thresholds(y_true, y_pred):
    iouts = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        iouts.append(compute_eval_metric_per_image(y_t, y_p))
    return np.mean(iouts)


def old_f_beta_metric(y_true, y_pred, beta=2, type="mean"):
    f_betas = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        f_betas.append(compute_eval_metric_per_image(y_t, y_p, "f{}".format(beta)))
    if type=='mean':
        return np.mean(f_betas)
    elif type=='per_image':
        return f_betas
    else:
        return NotImplementedError


def f_beta_metric(gt, prediction, beta=2):
    f_betas = []
    check_ids(gt, prediction)
    for image_id in gt['ImageId'].unique():
        y_t = get_overlayed_mask(gt.query('ImageId==@image_id'), ORIGINAL_SIZE, labeled=True)
        y_p = get_overlayed_mask(prediction.query('ImageId==@image_id'), ORIGINAL_SIZE, labeled=True)
        f_betas.append(compute_eval_metric_per_image(y_t, y_p, "f{}".format(beta)))
    return np.mean(f_betas)


def check_ids(gt, prediction):
    gt_ids = set(gt['ImageId'].unique())
    prediction_ids = set(prediction['ImageId'].unique())
    if gt_ids-prediction_ids != set():
        raise ValueError('Predictions for some images are missing')
    elif prediction_ids - gt_ids != set():
        raise ValueError('Prediction calculated for too many images')


def compute_f_beta_at(ious, threshold, beta):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + (beta**2) * fn + fp)
