import multiprocessing as mp

import logging
import pathlib
import random
import sys
from itertools import chain
from collections import Iterable
import math

import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib.pyplot as plt
from attrdict import AttrDict
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from steppy.base import Step, BaseTransformer
from steppy.utils import get_logger as get_steppy_logger
import yaml

from .masks import rle_from_mask, run_length_decoding

import os

NEPTUNE_CONFIG_PATH = os.environ.get('NEPTUNE_CONFIG_PATH',
                                     str(pathlib.Path(__file__).resolve().parents[1].parents[0] / 'neptune.yaml'))

logger = get_steppy_logger()


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        params = read_yaml().parameters
    else:
        params = ctx.params
    return params


def read_yaml(fallback_file=NEPTUNE_CONFIG_PATH):
    print("USING FALLBACK NEPTUNE CONFIG {}".format(fallback_file))
    with open(fallback_file) as f:
        config = yaml.load(f)
    return AttrDict(config)


def init_logger():
    logger = logging.getLogger('ships-detection')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def get_logger():
    return logging.getLogger('ships-detection')


def create_submission(image_ids, predictions):
    output = []
    for image_id, mask in zip(image_ids, predictions):
        for label_nr in range(1, mask.max() + 1):
            mask_label = mask == label_nr
            rle_encoded = ' '.join(str(rle) for rle in rle_from_mask(mask_label))
            output.append([image_id, rle_encoded])
        if mask.max() == 0:
            output.append([image_id, np.nan])

    submission = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels'])
    return submission


def get_ship_no_ship_ids(image_ids, prediction):
    ids_ship = [idx for idx, pred in zip(image_ids, prediction) if pred]
    ids_no_ship = [idx for idx, pred in zip(image_ids, prediction) if not pred]
    return ids_ship, ids_no_ship


def combine_two_stage_predictions(ids_no_ship, prediction_ship, ordered_ids):
    prediction_no_ship = pd.DataFrame({'ImageId': ids_no_ship})
    prediction_no_ship['EncodedPixels'] = np.nan

    prediction_ship.reset_index(drop=True, inplace=True)
    prediction_no_ship.reset_index(drop=True, inplace=True)

    prediction = pd.concat([prediction_ship, prediction_no_ship], axis=0)

    return prediction[prediction['ImageId'].isin(ordered_ids)]


def prepare_results(ground_truth, prediction, meta, f2, image_ids):
    f2_with_id = pd.DataFrame({'f2': f2, 'ImageId': image_ids})
    meta['ImageId'] = meta['id'] + '.jpg'
    scores_merged = pd.merge(ground_truth, prediction, on='ImageId', suffixes=['_gt', '_pred'])
    results = pd.merge(meta, scores_merged, on='ImageId')
    results = pd.merge(results, f2_with_id, on='ImageId')
    return results


def generate_data_frame_chunks(meta, chunk_size):
    n_rows = meta.shape[0]
    chunk_nr = math.ceil(n_rows / chunk_size)
    for i in tqdm(range(chunk_nr)):
        meta_chunk = meta.iloc[i * chunk_size:(i + 1) * chunk_size]
        yield meta_chunk


def generate_metadata(train_images_dir, masks_overlayed_dir, test_images_dir, annotation_file_name):
    metadata = {}
    annotations = pd.read_csv(annotation_file_name)
    for filename in tqdm(os.listdir(train_images_dir)):
        image_filepath = os.path.join(train_images_dir, filename)
        mask_filepath = os.path.join(masks_overlayed_dir, filename.split('.')[0])
        image_id = filename.split('.')[0]
        number_of_ships = get_number_of_ships(annotations.query('ImageId == @filename'))

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(mask_filepath)
        metadata.setdefault('is_train', []).append(1)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('number_of_ships', []).append(number_of_ships)
        metadata.setdefault('is_not_empty', []).append(int(number_of_ships != 0))

    for filename in tqdm(os.listdir(test_images_dir)):
        image_filepath = os.path.join(test_images_dir, filename)
        image_id = filename.split('.')[0]

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(None)
        metadata.setdefault('is_train', []).append(0)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('number_of_ships', []).append(None)
        metadata.setdefault('is_not_empty', []).append(None)

    return pd.DataFrame(metadata)


def get_number_of_ships(image_annotations):
    if image_annotations['EncodedPixels'].any():
        return len(image_annotations)
    else:
        return 0


def train_test_split_with_empty_fraction(df, empty_fraction, test_size, shuffle=True, random_state=1234):
    valid_empty_size = int(test_size * empty_fraction)
    valid_non_empty_size = int(test_size * (1.0 - empty_fraction))
    df_empty = df[df['is_not_empty'] == 0]
    df_non_empty = df[df['is_not_empty'] == 1]

    train_empty, valid_empty = train_test_split(df_empty,
                                                test_size=valid_empty_size,
                                                shuffle=shuffle,
                                                random_state=random_state)
    train_non_empty, valid_non_empty = train_test_split(df_non_empty,
                                                        test_size=valid_non_empty_size,
                                                        shuffle=shuffle,
                                                        random_state=random_state)

    train = pd.concat([train_empty, train_non_empty], axis=0)
    valid = pd.concat([valid_empty, valid_non_empty], axis=0)

    return train, valid


def train_test_split_with_empty_fraction_with_groups(df,
                                                     groups,
                                                     empty_fraction,
                                                     test_size,
                                                     shuffle=True, random_state=1234):
    cv = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)

    for train_inds, test_inds in cv.split(df.values, groups=groups.values):
        train, test = df.iloc[train_inds], df.iloc[test_inds]
        break

    empty_train, empty_test = train[train['is_not_empty'] == 0], test[test['is_not_empty'] == 0]
    non_empty_train, non_empty_test = train[train['is_not_empty'] == 1], test[test['is_not_empty'] == 1]

    test_empty_size = int(test_size * empty_fraction)
    test_non_empty_size = int(test_size * (1.0 - empty_fraction))

    empty_test = empty_test.sample(test_empty_size, random_state=random_state)
    non_empty_test = non_empty_test.sample(test_non_empty_size, random_state=random_state)

    train = pd.concat([empty_train, non_empty_train], axis=0).sample(frac=1, random_state=random_state)
    test = pd.concat([empty_test, non_empty_test], axis=0)

    if shuffle:
        train = train.sample(frac=1, random_state=random_state)
        test = test.sample(frac=1, random_state=random_state)

    return train, test


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(X, theta=1.0, axis=None):
    """
    https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def get_crop_pad_sequence(vertical, horizontal):
    top = int(vertical / 2)
    bottom = vertical - top
    right = int(horizontal / 2)
    left = horizontal - right
    return (top, right, bottom, left)


def get_list_of_image_predictions(batch_predictions):
    image_predictions = []
    for batch_pred in batch_predictions:
        image_predictions.extend(list(batch_pred))
    return image_predictions


def relabel(img):
    relabeled_img = np.zeros_like(img)
    for i, k in enumerate(np.unique(img)):
        if k == 0:
            continue
        else:
            relabeled_img = np.where(img == k, i, relabeled_img)
    return relabeled_img


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_list(images=None, labels=None):
    images = [] if not images else images
    labels = [] if not labels else labels

    n_img = len(images)
    n_lab = len(labels)
    n = n_lab + n_img
    fig, axs = plt.subplots(1, n, figsize=(16, 12))
    for i, image in enumerate(images):
        axs[i].imshow(image)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    for j, label in enumerate(labels):
        axs[n_img + j].imshow(label, cmap='nipy_spectral')
        axs[n_img + j].set_xticks([])
        axs[n_img + j].set_yticks([])
    plt.show()


def plot_results_for_id(results, idx):
    results_per_image = results[results['ImageId'] == idx]
    file_path_image = results_per_image.file_path_image.values[0]
    image = np.array(Image.open(file_path_image)).astype(np.uint8)
    ground_truth_image = np.zeros((image.shape[:2]))
    prediction_image = np.zeros((image.shape[:2]))

    ground_truth_rle = results_per_image.EncodedPixels_gt.unique()
    prediction_rle = results_per_image.EncodedPixels_pred.unique()

    for i, gt in enumerate(ground_truth_rle):
        if isinstance(gt, float):
            continue
        obj_mask = run_length_decoding(gt)
        ground_truth_image = np.where(obj_mask, i + 1, ground_truth_image)

    for i, pred in enumerate(prediction_rle):
        if isinstance(pred, float):
            continue
        obj_mask = run_length_decoding(pred)
        prediction_image = np.where(obj_mask, i + 1, prediction_image)

    plot_list(images=[image], labels=[ground_truth_image, prediction_image])


class FineTuneStep(Step):
    def __init__(self,
                 transformer,
                 name=None,
                 experiment_directory=None,
                 output_directory=None,
                 input_data=None,
                 input_steps=None,
                 adapter=None,

                 is_fittable=True,
                 force_fitting=True,
                 fine_tuning=False,

                 persist_output=False,
                 cache_output=False,
                 load_persisted_output=False):
        super().__init__(name=name,
                         transformer=transformer,
                         experiment_directory=experiment_directory,
                         output_directory=output_directory,
                         input_data=input_data,
                         input_steps=input_steps,
                         adapter=adapter,
                         is_fittable=is_fittable,
                         force_fitting=force_fitting,
                         cache_output=cache_output,
                         persist_output=persist_output,
                         load_persisted_output=load_persisted_output)
        self.fine_tuning = fine_tuning

    def _fit_transform_operation(self, step_inputs):
        if self.is_fittable:
            if self.transformer_is_persisted:
                if self.force_fitting and self.fine_tuning:
                    raise ValueError('only one of force_fitting or fine_tuning can be True')
                elif self.force_fitting:
                    logger.info('Step {}, fitting and transforming...'.format(self.name))
                    step_output_data = self.transformer.fit_transform(**step_inputs)
                    logger.info('Step {}, persisting transformer to the {}'
                                .format(self.name, self.experiment_directory_transformers_step))
                    self.transformer.persist(self.experiment_directory_transformers_step)
                elif self.fine_tuning:
                    logger.info('Step {}, loading transformer from the {}'
                                .format(self.name, self.experiment_directory_transformers_step))
                    self.transformer.load(self.experiment_directory_transformers_step)
                    logger.info('Step {}, transforming...'.format(self.name))
                    step_output_data = self.transformer.fit_transform(**step_inputs)
                    self.transformer.persist(self.experiment_directory_transformers_step)
                else:
                    logger.info('Step {}, loading transformer from the {}'
                                .format(self.name, self.experiment_directory_transformers_step))
                    self.transformer.load(self.experiment_directory_transformers_step)
                    logger.info('Step {}, transforming...'.format(self.name))
                    step_output_data = self.transformer.transform(**step_inputs)
            else:
                logger.info('Step {}, fitting and transforming...'.format(self.name))
                step_output_data = self.transformer.fit_transform(**step_inputs)
                logger.info('Step {}, persisting transformer to the {}'
                            .format(self.name, self.experiment_directory_transformers_step))
                self.transformer.persist(self.experiment_directory_transformers_step)
        else:
            logger.info('Step {}, transforming...'.format(self.name))
            step_output_data = self.transformer.transform(**step_inputs)

        if self.cache_output:
            logger.info('Step {}, caching output to the {}'
                        .format(self.name, self.experiment_directory_output_step))
            self.output = step_output_data
        if self.persist_output:
            logger.info('Step {}, persisting output to the {}'
                        .format(self.name, self.experiment_directory_output_step))
            self._persist_output(step_output_data, self.experiment_directory_output_step)
        return step_output_data


def make_apply_transformer(func, output_name='output', apply_on=None, n_threads=1):
    class StaticApplyTransformer(BaseTransformer):
        def transform(self, *args, **kwargs):
            self.check_input(*args, **kwargs)
            if not apply_on:
                iterator = list(zip(*args, *kwargs.values()))
            else:
                iterator = list(zip(*args, *[kwargs[key] for key in apply_on]))

            n_jobs = np.minimum(n_threads, len(iterator))
            with mp.pool.ThreadPool(n_jobs) as executor:
                output = list(tqdm(executor.imap(lambda p: func(*p), iterator), total=len(iterator)))
            return {output_name: output}

        @staticmethod
        def check_input(*args, **kwargs):
            if len(args) and len(kwargs) == 0:
                raise Exception('Input must not be empty')

            arg_length = None
            for arg in chain(args, kwargs.values()):
                if not isinstance(arg, Iterable):
                    raise Exception('All inputs must be iterable')
                arg_length_loc = None
                try:
                    arg_length_loc = len(arg)
                except:
                    pass
                if arg_length_loc is not None:
                    if arg_length is None:
                        arg_length = arg_length_loc
                    elif arg_length_loc != arg_length:
                        raise Exception('All inputs must be the same length')

        @staticmethod
        def get_arg_length(*args, **kwargs):
            arg_length = None
            for arg in chain(args, kwargs.values()):
                if arg_length is None:
                    try:
                        arg_length = len(arg)
                    except:
                        pass
                if arg_length is not None:
                    return arg_length

    return StaticApplyTransformer()


class OneCycle(object):
    """
    In paper (https://arxiv.org/pdf/1803.09820.pdf), author suggests to do one cycle during
    whole run with 2 steps of equal length. During first step, increase the learning rate
    from lower learning rate to higher learning rate. And in second step, decrease it from
    higher to lower learning rate. This is Cyclic learning rate policy. Author suggests one
    addition to this. - During last few hundred/thousand iterations of cycle reduce the
    learning rate to 1/100th or 1/1000th of the lower learning rate.
    Also, Author suggests that reducing momentum when learning rate is increasing. So, we make
    one cycle of momentum also with learning rate - Decrease momentum when learning rate is
    increasing and increase momentum when learning rate is decreasing.
    Args:

        nb              Total number of iterations including all epochs
        max_lr          The optimum learning rate. This learning rate will be used as highest
                        learning rate. The learning rate will fluctuate between max_lr to
                        max_lr/div and then (max_lr/div)/div.
        momentum_vals   The maximum and minimum momentum values between which momentum will
                        fluctuate during cycle.
                        Default values are (0.95, 0.85)
        prcnt           The percentage of cycle length for which we annihilate learning rate
                        way below the lower learnig rate.
                        The default value is 10
        div             The division factor used to get lower boundary of learning rate. This
                        will be used with max_lr value to decide lower learning rate boundary.
                        This value is also used to decide how much we annihilate the learning
                        rate below lower learning rate.
                        The default value is 10.
    """

    def __init__(self, nb, max_lr, optimizer=None, momentum_vals=(0.95, 0.85), prcnt=10, div=10):
        self.nb = nb
        self.div = div
        self.step_len = int(self.nb * (1 - prcnt / 100) / 2)
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt
        self.iteration = 0
        self.lrs = []
        self.moms = []
        self.optimizer = optimizer

    def batch_step(self):
        if self.optimizer is None:
            raise ValueError("""
            Can you have to provide an optimizer otherwise 
            you can only use calc_lr anc calc_mom methods""")

        lr, mom = self.calc()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['mom'] = mom

        return lr, mom

    def calc(self):
        self.iteration += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        return (lr, mom)

    def calc_lr(self):
        if self.iteration == self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr / self.div)
            return self.high_lr / self.div
        if self.iteration > 2 * self.step_len:
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            lr = self.high_lr * (1 - 0.99 * ratio) / self.div
        elif self.iteration > self.step_len:
            ratio = 1 - (self.iteration - self.step_len) / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else:
            ratio = self.iteration / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr

    def calc_mom(self):
        if self.iteration == self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        if self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration - self.step_len) / self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:
            ratio = self.iteration / self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom
