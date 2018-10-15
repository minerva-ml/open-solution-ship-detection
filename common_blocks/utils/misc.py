import logging
import os
import pathlib
import random
import sys
from itertools import chain
from collections import Iterable
import math

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from attrdict import AttrDict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from steppy.base import Step, BaseTransformer
from steppy.utils import get_logger as get_steppy_logger
import yaml

from .masks import rle_from_mask
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
            output.append([image_id, None])

    submission = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels'])
    return submission


def get_ship_no_ship_ids(image_ids, prediction):
    ids_ship = [idx for idx, pred in zip(image_ids, prediction) if pred]
    ids_no_ship = [idx for idx, pred in zip(image_ids, prediction) if not pred]
    return ids_ship, ids_no_ship


def combine_two_stage_predictions(ids_no_ship, prediction_ship, ordered_ids):
    prediction_no_ship = pd.DataFrame({'ImageId': ids_no_ship})
    prediction_no_ship['EncodedPixels'] = None

    prediction_ship.reset_index(drop=True, inplace=True)
    prediction_no_ship.reset_index(drop=True, inplace=True)

    prediction = pd.concat([prediction_ship, prediction_no_ship], axis=0)

    return prediction[prediction['ImageId'].isin(ordered_ids)]


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


def make_apply_transformer(func, output_name='output', apply_on=None):
    class StaticApplyTransformer(BaseTransformer):
        def transform(self, *args, **kwargs):
            self.check_input(*args, **kwargs)

            if not apply_on:
                iterator = zip(*args, *kwargs.values())
            else:
                iterator = zip(*args, *[kwargs[key] for key in apply_on])

            output = []
            for func_args in tqdm(iterator, total=self.get_arg_length(*args, **kwargs)):
                output.append(func(*func_args))
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
