import os
import shutil

import pandas as pd

from .metrics import f_beta_metric
from . import pipeline_config as cfg
from .pipelines import PIPELINES
from .utils import NeptuneContext, init_logger, read_gt_subset, create_submission, \
    generate_metadata, set_seed, clean_memory, generate_data_frame_chunks
from .preparation import train_valid_split, overlay_masks

LOGGER = init_logger()
CTX = NeptuneContext()
PARAMS = CTX.params
set_seed(cfg.SEED)


class PipelineManager:
    def prepare_masks(self, dev_mode):
        prepare_masks(dev_mode)

    def prepare_metadata(self):
        prepare_metadata()

    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode)

    def evaluate(self, pipeline_name, dev_mode, chunk_size):
        evaluate(pipeline_name, dev_mode, chunk_size)

    def predict(self, pipeline_name, dev_mode, submit_predictions, chunk_size):
        predict(pipeline_name, dev_mode, submit_predictions, chunk_size)

    def make_submission(self, submission_filepath):
        make_submission(submission_filepath)


def prepare_masks(dev_mode):
    LOGGER.info('overlaying masks')
    overlay_masks(annotation_file_name=PARAMS.annotation_file, target_dir=PARAMS.masks_overlayed_dir, dev_mode=dev_mode)


def prepare_metadata():
    LOGGER.info('creating metadata')
    meta = generate_metadata(train_images_dir=PARAMS.train_images_dir,
                             masks_overlayed_dir=PARAMS.masks_overlayed_dir,
                             test_images_dir=PARAMS.test_images_dir,
                             annotation_file_name=PARAMS.annotation_file
                             )
    meta.to_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'), index=None)


def train(pipeline_name, dev_mode):
    LOGGER.info('training')
    if bool(PARAMS.overwrite) and os.path.isdir(PARAMS.experiment_dir):
        shutil.rmtree(PARAMS.experiment_dir)

    meta = pd.read_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'))
    meta_train = meta[meta['is_train'] == 1]
    evaluation_size = PARAMS.evaluation_size
    validation_size = PARAMS.validation_size
    meta_train_split, meta_valid_split = train_valid_split(meta_train, evaluation_size, validation_size,
                                                           random_state=cfg.SEED)

    if dev_mode:
        meta_train_split = meta_train_split.sample(PARAMS.dev_mode_size, random_state=cfg.SEED)
        meta_valid_split = meta_valid_split.sample(int(PARAMS.dev_mode_size / 2), random_state=cfg.SEED)

    data = {'input': {'meta': meta_train_split,},
            'callback_input': {'meta_valid': meta_valid_split}
            }

    pipeline = PIPELINES[pipeline_name]['train'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


def evaluate(pipeline_name, dev_mode, chunk_size):
    LOGGER.info('evaluating')
    meta = pd.read_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'))
    meta_train = meta[meta['is_train'] == 1]

    evaluation_size = PARAMS.evaluation_size
    meta_train_split, meta_valid_split = train_valid_split(meta_train, evaluation_size, validation_size=1,
                                                           random_state=cfg.SEED)

    if dev_mode:
        meta_valid_split = meta_valid_split.sample(PARAMS.dev_mode_size, random_state=cfg.SEED)

    pipeline = PIPELINES[pipeline_name]['inference'](config=cfg.SOLUTION_CONFIG)
    prediction = generate_submission(meta_valid_split, pipeline, chunk_size)
    gt = read_gt_subset(PARAMS.annotation_file, meta_valid_split[cfg.ID_COLUMNS[0]]+'.jpg')
    f2_score = f_beta_metric(gt, prediction)
    LOGGER.info('F2 score on validation is {}'.format(f2_score))
    CTX.channel_send('f2', 0, f2_score)


def predict(pipeline_name, dev_mode, submit_predictions, chunk_size):
    LOGGER.info('predicting')
    meta = pd.read_csv(os.path.join(PARAMS.meta_dir, 'metadata.csv'))
    meta_test = meta[meta['is_train'] == 0]

    if dev_mode:
        meta_test = meta_test.sample(PARAMS.dev_mode_size, random_state=cfg.SEED)

    pipeline = PIPELINES[pipeline_name]['inference'](config=cfg.SOLUTION_CONFIG)

    submission = generate_submission(meta_test, pipeline, chunk_size)

    submission_filepath = os.path.join(PARAMS.experiment_dir, 'submission.csv')

    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    LOGGER.info('submission saved to {}'.format(submission_filepath))
    LOGGER.info('submission head \n\n{}'.format(submission.head()))

    if submit_predictions:
        make_submission(submission_filepath)


def make_submission(submission_filepath):
    LOGGER.info('Making Kaggle submit...')
    os.system('kaggle competitions submit -c airbus-ship-detection -f {} -m {}'.format(submission_filepath,
                                                                                       PARAMS.kaggle_message))
    LOGGER.info('Kaggle submit completed')


def generate_submission(meta_data, pipeline, chunk_size):
    if chunk_size is not None:
        return _generate_submission_in_chunks(meta_data, pipeline, chunk_size)
    else:
        return _generate_submission(meta_data, pipeline)


def _generate_submission(meta_data, pipeline):
    prediction = _generate_prediction(meta_data, pipeline)
    submission = create_submission(meta_data[cfg.ID_COLUMNS[0]]+'.jpg', prediction)
    return submission


def _generate_submission_in_chunks(meta_data, pipeline, chunk_size):
    submissions = []
    for meta_chunk in generate_data_frame_chunks(meta_data, chunk_size):
        prediction_chunk = _generate_prediction(meta_chunk, pipeline)
        submission_chunk = create_submission(meta_chunk[cfg.ID_COLUMNS[0]]+'.jpg', prediction_chunk)
        submissions.append(submission_chunk)
    submission = pd.concat(submissions)
    return submission


def _generate_prediction(meta_data, pipeline):
    data = {'input': {'meta': meta_data,
                      },
            'callback_input': {'meta_valid': None
                               }
            }
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']
    return y_pred
