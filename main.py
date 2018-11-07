import os
import shutil
from functools import partial

import neptune
import numpy as np
import pandas as pd
from attrdict import AttrDict
from steppy.adapter import Adapter, E
from steppy.base import IdentityOperation, Step

from common_blocks import augmentation as aug
from common_blocks import metrics
from common_blocks import models
from common_blocks import pipelines
from common_blocks import postprocessing
from common_blocks.utils import io, misc

CTX = neptune.Context()
LOGGER = misc.init_logger()

#    ______   ______   .__   __.  _______  __    _______      _______.
#   /      | /  __  \  |  \ |  | |   ____||  |  /  _____|    /       |
#  |  ,----'|  |  |  | |   \|  | |  |__   |  | |  |  __     |   (----`
#  |  |     |  |  |  | |  . `  | |   __|  |  | |  | |_ |     \   \
#  |  `----.|  `--'  | |  |\   | |  |     |  | |  |__| | .----)   |
#   \______| \______/  |__| \__| |__|     |__|  \______| |_______/
#

EXPERIMENT_DIR = '/output/experiment'
CLONE_EXPERIMENT_DIR_FROM = ''  # When running eval in the cloud specify this as for example /input/SHIP-14/output/experiment
OVERWRITE_EXPERIMENT_DIR = False

DEV_MODE = False
USE_TTA = True
INFERENCE_WITH_SHIP_NO_SHIP = True

if OVERWRITE_EXPERIMENT_DIR and os.path.isdir(EXPERIMENT_DIR):
    shutil.rmtree(EXPERIMENT_DIR)
if CLONE_EXPERIMENT_DIR_FROM != '':
    if os.path.exists(EXPERIMENT_DIR):
        shutil.rmtree(EXPERIMENT_DIR)
    shutil.copytree(CLONE_EXPERIMENT_DIR_FROM, EXPERIMENT_DIR)

if CTX.params.__class__.__name__ == 'OfflineContextParams':
    PARAMS = misc.read_yaml().parameters
else:
    PARAMS = CTX.params

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CHUNK_SIZE_SHIP_NO_SHIP = 2500
CHUNK_SIZE_SEGMENTATION = 2500
SEED = 1234
ID_COLUMN = 'id'
ID_BIG_IMAGE = 'BigImageId'
IS_NOT_EMPTY_COLUMN = 'is_not_empty'
X_COLUMN = 'file_path_image'
Y_COLUMN = 'file_path_mask'

CONFIG = AttrDict({
    'execution': {'experiment_dir': EXPERIMENT_DIR,
                  'num_workers': PARAMS.num_workers,
                  'num_threads': PARAMS.num_threads
                  },
    'general': {'img_H-W': (PARAMS.image_h, PARAMS.image_w),
                'loader_mode': PARAMS.loader_mode,
                'num_classes': 2,
                'original_size': (768, 768),
                },
    'meta_reader': {
        'segmentation_network': {'x_columns': [X_COLUMN],
                                 'y_columns': [Y_COLUMN, IS_NOT_EMPTY_COLUMN],
                                 },
        'ship_no_ship_network': {'x_columns': [X_COLUMN],
                                 'y_columns': [IS_NOT_EMPTY_COLUMN],
                                 },
    },
    'loaders': {'resize': {'dataset_params': {'h': PARAMS.image_h,
                                              'w': PARAMS.image_w,
                                              'sns_h': PARAMS.sns_image_h,
                                              'sns_w': PARAMS.sns_image_w,

                                              'image_source': PARAMS.image_source,
                                              'target_format': PARAMS.target_format,
                                              'empty_fraction': PARAMS.training_sampler_empty_fraction,
                                              'sample_size': PARAMS.training_sampler_size,
                                              'sns_empty_fraction': PARAMS.sns_training_sampler_empty_fracion,
                                              'MEAN': MEAN,
                                              'STD': STD
                                              },
                           'loader_params': {'training': {'batch_size': PARAMS.batch_size_train,
                                                          'shuffle': False,
                                                          'num_workers': PARAMS.num_workers,
                                                          'pin_memory': PARAMS.pin_memory
                                                          },
                                             'inference': {'batch_size': PARAMS.batch_size_inference,
                                                           'shuffle': False,
                                                           'num_workers': PARAMS.num_workers,
                                                           'pin_memory': PARAMS.pin_memory
                                                           },
                                             },

                           'augmentation_params': {'image_augment_train': aug.intensity_seq,
                                                   'image_augment_with_target_train': aug.resize_seq(
                                                       resize_target_size=PARAMS.resize_target_size),
                                                   'image_augment_inference': aug.resize_to_fit_net(
                                                       resize_target_size=PARAMS.resize_target_size),
                                                   'image_augment_with_target_inference': aug.resize_to_fit_net(
                                                       resize_target_size=PARAMS.resize_target_size)
                                                   },
                           },
                'resize_tta': {'dataset_params': {'h': PARAMS.image_h,
                                                  'w': PARAMS.image_w,
                                                  'image_source': PARAMS.image_source,
                                                  'target_format': PARAMS.target_format,
                                                  'empty_fraction': PARAMS.training_sampler_empty_fraction,
                                                  'sample_size': PARAMS.training_sampler_size,
                                                  'MEAN': MEAN,
                                                  'STD': STD
                                                  },
                               'loader_params': {'training': {'batch_size': PARAMS.batch_size_train,
                                                              'shuffle': False,
                                                              'num_workers': PARAMS.num_workers,
                                                              'pin_memory': PARAMS.pin_memory
                                                              },
                                                 'inference': {'batch_size': PARAMS.batch_size_inference,
                                                               'shuffle': False,
                                                               'num_workers': PARAMS.num_workers,
                                                               'pin_memory': PARAMS.pin_memory
                                                               },
                                                 },

                               'augmentation_params': {
                                   'image_augment_inference': aug.resize_to_fit_net(
                                       resize_target_size=PARAMS.resize_target_size),
                                   'image_augment_with_target_inference': aug.resize_to_fit_net(
                                       resize_target_size=PARAMS.resize_target_size),
                                   'tta_transform': aug.test_time_augmentation_transform
                               },
                               },
                },
    'model': {
        'segmentation_network': {
            'architecture_config': {'model_params': {'in_channels': PARAMS.image_channels,
                                                     'out_channels': PARAMS.network_output_channels,
                                                     'architecture': PARAMS.architecture,
                                                     'encoder': PARAMS.encoder,
                                                     'activation': PARAMS.network_activation,
                                                     },
                                    'optimizer_params': {'lr': PARAMS.lr,
                                                         },
                                    'regularizer_params': {'regularize': True,
                                                           'weight_decay_conv2d': PARAMS.l2_reg_conv,
                                                           },
                                    'weights_init': {'function': 'xavier',
                                                     },
                                    },
            'training_config': {'epochs': PARAMS.epochs_nr,
                                'shuffle': True,
                                'batch_size': PARAMS.batch_size_train,
                                'fine_tuning': PARAMS.fine_tuning,
                                },
            'callbacks_config': {'model_checkpoint': {
                'filepath': os.path.join(EXPERIMENT_DIR, 'checkpoints', 'segmentation_network', 'best.torch'),
                'epoch_every': 1,
                'metric_name': PARAMS.validation_metric_name,
                'minimize': PARAMS.minimize_validation_metric},
                "one_cycle_scheduler": {

                    "enabled": PARAMS.use_one_cycle,
                    "number_of_batches_per_full_cycle": PARAMS.one_cycle_number_of_batches_per_full_cycle,
                    "max_lr": PARAMS.one_cycle_max_lr,
                    "momentum_range": (0.95, 0.8),
                    "prcnt_annihilate": 10,
                    "div": 10
                },
                'exponential_lr_scheduler': {'gamma': PARAMS.gamma,
                                             'epoch_every': 1},
                'reduce_lr_on_plateau_scheduler': {'metric_name': PARAMS.validation_metric_name,
                                                   'minimize': PARAMS.minimize_validation_metric,
                                                   'reduce_factor': PARAMS.reduce_factor,
                                                   'reduce_patience': PARAMS.reduce_patience,
                                                   'min_lr': PARAMS.min_lr},
                'training_monitor': {'batch_every': 1,
                                     'epoch_every': 1},
                'experiment_timing': {'batch_every': 10,
                                      'epoch_every': 1},
                'validation_monitor': {'epoch_every': 1,
                                       'data_dir': PARAMS.train_images_dir,
                                       'loader_mode': PARAMS.loader_mode},
                'neptune_monitor': {'model_name': 'network',
                                    'image_nr': 16,
                                    'image_resize': 1.0,
                                    'image_every': 1},
                'early_stopping': {'patience': PARAMS.patience,
                                   'metric_name': PARAMS.validation_metric_name,
                                   'minimize': PARAMS.minimize_validation_metric},
            }
        },

        'ship_no_ship_network': {
            'architecture_config': {'model_params': {'architecture': PARAMS.sns_architecture,
                                                     'activation': 'sigmoid'},
                                    'optimizer_params': {'lr': PARAMS.sns_lr,
                                                         },
                                    'regularizer_params': {'regularize': True,
                                                           'weight_decay_conv2d': PARAMS.sns_l2_reg_conv,
                                                           },
                                    'weights_init': {'function': 'xavier',
                                                     },
                                    },
            'training_config': {'epochs': PARAMS.sns_epochs_nr,
                                'shuffle': True,
                                'batch_size': PARAMS.sns_batch_size_train,
                                'fine_tuning': PARAMS.fine_tuning,
                                },
            'callbacks_config': {'model_checkpoint': {
                'filepath': os.path.join(EXPERIMENT_DIR, 'checkpoints', 'ship_no_ship_network', 'best.torch'),
                'epoch_every': 1,
                'metric_name': PARAMS.sns_validation_metric_name,
                'minimize': PARAMS.sns_minimize_validation_metric
            },
                "one_cycle_scheduler": {

                    "enabled": PARAMS.sns_use_one_cycle,
                    "number_of_batches_per_full_cycle": PARAMS.sns_one_cycle_number_of_batches_per_full_cycle,
                    "max_lr": PARAMS.sns_one_cycle_max_lr,
                    "momentum_range": (0.95, 0.7),
                    "prcnt_annihilate": 10,
                    "div": 10
                },
                'exponential_lr_scheduler': {'gamma': PARAMS.gamma,
                                             'epoch_every': 1},
                'reduce_lr_on_plateau_scheduler': {'metric_name': PARAMS.sns_validation_metric_name,
                                                   'minimize': PARAMS.sns_minimize_validation_metric,
                                                   'reduce_factor': PARAMS.reduce_factor,
                                                   'reduce_patience': PARAMS.reduce_patience,
                                                   'min_lr': PARAMS.min_lr},
                'training_monitor': {'batch_every': 10,
                                     'epoch_every': 1},
                'experiment_timing': {'batch_every': 10,
                                      'epoch_every': 1},
                'validation_monitor': {'epoch_every': 1,
                                       'data_dir': PARAMS.train_images_dir,
                                       'loader_mode': PARAMS.loader_mode},
                'neptune_monitor': {'model_name': 'network',
                                    'image_nr': 16,
                                    'image_resize': 1.0,
                                    'image_every': 1},

                'early_stopping': {'patience': PARAMS.patience,
                                   'metric_name': PARAMS.validation_metric_name,
                                   'minimize': PARAMS.minimize_validation_metric},
            }
        },
    },
    'tta_generator': {'flip_ud': True,
                      'flip_lr': True,
                      'rotation': True,
                      'color_shift_runs': False},
    'tta_aggregator': {'tta_inverse_transform': aug.test_time_augmentation_inverse_transform,
                       'method': PARAMS.tta_aggregation_method,
                       'nthreads': PARAMS.num_threads
                       },
    'thresholder': {'threshold_masks': PARAMS.threshold_masks,
                    'threshold_ship_no_ship': PARAMS.sns_threshold,
                    },
})


#  .______    __  .______    _______  __       __  .__   __.  _______     _______.
#  |   _  \  |  | |   _  \  |   ____||  |     |  | |  \ |  | |   ____|   /       |
#  |  |_)  | |  | |  |_)  | |  |__   |  |     |  | |   \|  | |  |__     |   (----`
#  |   ___/  |  | |   ___/  |   __|  |  |     |  | |  . `  | |   __|     \   \
#  |  |      |  | |  |      |  |____ |  `----.|  | |  |\   | |  |____.----)   |
#  | _|      |__| | _|      |_______||_______||__| |__| \__| |_______|_______/
#

def ship_no_ship_pipeline(config, suffix='_ship_no_ship', train_mode=True):
    if train_mode:
        preprocessing = pipelines.preprocessing_binary_train(config, model_name='ship_no_ship_network', suffix=suffix)
    else:
        preprocessing = pipelines.preprocessing_binary_inference(config, model_name='ship_no_ship_network',
                                                                 suffix=suffix)
    preprocessing.set_parameters_upstream({'is_fittable': False})

    sns_network = misc.FineTuneStep(name='ship_no_ship_network',
                                    transformer=models.BinaryModel(**config.model['ship_no_ship_network']),
                                    input_steps=[preprocessing],
                                    )

    sns_network.set_mode_train()
    sns_network.set_parameters_upstream({'experiment_directory': config.execution.experiment_dir,
                                         })
    sns_network.force_fitting = False
    sns_network.fine_tuning = config.model.segmentation_network.training_config.fine_tuning

    if train_mode:
        return sns_network
    else:
        class_prediction = Step(name='class_prediction',
                                transformer=misc.make_apply_transformer(
                                    partial(postprocessing.get_class,
                                            threshold=config.thresholder.threshold_ship_no_ship),
                                    output_name='classes',
                                    apply_on=['predictions']),
                                input_steps=[sns_network],
                                adapter=Adapter({'predictions': E(sns_network.name, 'ship_no_ship_prediction'),
                                                 }),
                                is_fittable=False
                                )
        return class_prediction


def train_segmentation_pipeline(config):
    preprocessing = pipelines.preprocessing_train(config, model_name='segmentation_network')

    segmentation_network = misc.FineTuneStep(name='segmentation_network',
                                             transformer=models.SegmentationModel(
                                                 **config.model['segmentation_network']),
                                             input_data=['callback_input'],
                                             input_steps=[preprocessing],
                                             adapter=Adapter({'datagen': E(preprocessing.name, 'datagen'),
                                                              'validation_datagen': E(preprocessing.name,
                                                                                      'validation_datagen'),
                                                              'meta_valid': E('callback_input', 'meta_valid'),
                                                              }))

    segmentation_network.set_mode_train()
    segmentation_network.set_parameters_upstream({'experiment_directory': config.execution.experiment_dir,
                                                  })
    segmentation_network.force_fitting = False
    segmentation_network.fine_tuning = config.model.segmentation_network.training_config.fine_tuning
    return segmentation_network


def inference_segmentation_pipeline(config):
    if config.general.loader_mode == 'resize_and_pad':
        size_adjustment_function = partial(postprocessing.crop_image, target_size=config.general.original_size)
    elif config.general.loader_mode == 'resize' or config.general.loader_mode == 'stacking':
        size_adjustment_function = partial(postprocessing.resize_image, target_size=config.general.original_size)
    else:
        raise NotImplementedError

    if USE_TTA:
        preprocessing, tta_generator = pipelines.preprocessing_inference_tta(config, model_name='segmentation_network')

        segmentation_network = Step(name='segmentation_network',
                                    transformer=models.SegmentationModel(**config.model['segmentation_network']),
                                    input_steps=[preprocessing])

        tta_aggregator = pipelines.aggregator('tta_aggregator', segmentation_network,
                                              tta_generator=tta_generator,
                                              config=config.tta_aggregator)

        prediction_renamed = Step(name='prediction_renamed',

                                  transformer=IdentityOperation(),
                                  input_steps=[tta_aggregator],
                                  adapter=Adapter({'mask_prediction': E(tta_aggregator.name, 'aggregated_prediction')
                                                   }))

        mask_resize = Step(name='mask_resize',
                           transformer=misc.make_apply_transformer(size_adjustment_function,
                                                                   output_name='resized_images',
                                                                   apply_on=['images'],
                                                                   n_threads=config.execution.num_threads,
                                                                   ),
                           input_steps=[prediction_renamed],
                           adapter=Adapter({'images': E(prediction_renamed.name, 'mask_prediction'),
                                            }))
    else:

        preprocessing = pipelines.preprocessing_inference(config, model_name='segmentation_network')

        segmentation_network = misc.FineTuneStep(name='segmentation_network',
                                                 transformer=models.SegmentationModel(
                                                     **config.model['segmentation_network']),
                                                 input_steps=[preprocessing],
                                                 )

        mask_resize = Step(name='mask_resize',

                           transformer=misc.make_apply_transformer(size_adjustment_function,
                                                                   output_name='resized_images',
                                                                   apply_on=['images'],
                                                                   n_threads=config.execution.num_threads,
                                                                   ),
                           input_steps=[segmentation_network],
                           adapter=Adapter({'images': E(segmentation_network.name, 'mask_prediction'),
                                            }),
                           )

    binarizer = Step(name='binarizer',
                     transformer=misc.make_apply_transformer(
                         partial(postprocessing.binarize, threshold=config.thresholder.threshold_masks),
                         output_name='binarized_images',
                         apply_on=['images'],
                         n_threads=config.execution.num_threads
                     ),
                     input_steps=[mask_resize],
                     adapter=Adapter({'images': E(mask_resize.name, 'resized_images'),
                                      }))

    labeler = Step(name='labeler',
                   transformer=misc.make_apply_transformer(postprocessing.label,
                                                           output_name='labeled_images',
                                                           apply_on=['images'],
                                                           n_threads=config.execution.num_threads,
                                                           ),
                   input_steps=[binarizer],
                   adapter=Adapter({'images': E(binarizer.name, 'binarized_images'),
                                    }))
    mask_postprocessing = Step(name='mask_postprocessing',
                               transformer=misc.make_apply_transformer(postprocessing.mask_postprocessing,
                                                                       output_name='labeled_images',
                                                                       apply_on=['images'],
                                                                       n_threads=config.execution.num_threads,
                                                                       ),
                               input_steps=[labeler],
                               adapter=Adapter({'images': E(labeler.name, 'labeled_images'),
                                                }))

    mask_postprocessing.set_mode_inference()
    mask_postprocessing.set_parameters_upstream({'experiment_directory': config.execution.experiment_dir,
                                                 'is_fittable': False
                                                 })
    segmentation_network.is_fittable = True
    return mask_postprocessing


#   __________   ___  _______   ______  __    __  .___________. __    ______   .__   __.
#  |   ____\  \ /  / |   ____| /      ||  |  |  | |           ||  |  /  __  \  |  \ |  |
#  |  |__   \  V  /  |  |__   |  ,----'|  |  |  | `---|  |----`|  | |  |  |  | |   \|  |
#  |   __|   >   <   |   __|  |  |     |  |  |  |     |  |     |  | |  |  |  | |  . `  |
#  |  |____ /  .  \  |  |____ |  `----.|  `--'  |     |  |     |  | |  `--'  | |  |\   |
#  |_______/__/ \__\ |_______| \______| \______/      |__|     |__|  \______/  |__| \__|
#

def train_ship_no_ship():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    meta_train = add_big_image_id(meta_train)
    meta_train_split, meta_valid_split = misc.train_test_split_with_empty_fraction_with_groups(meta_train,
                                                                                               groups=meta_train[
                                                                                                   ID_BIG_IMAGE],
                                                                                               empty_fraction=PARAMS.evaluation_empty_fraction,
                                                                                               test_size=PARAMS.evaluation_size,
                                                                                               shuffle=True,
                                                                                               random_state=SEED)

    meta_train_split = meta_train_split.sample(frac=1, random_state=SEED)
    meta_valid_split = meta_valid_split.sample(frac=1, random_state=SEED)

    if DEV_MODE:
        meta_train_split = meta_train_split.sample(PARAMS.dev_mode_size, random_state=SEED)
        meta_valid_split = meta_valid_split.sample(int(PARAMS.dev_mode_size / 2), random_state=SEED)

    data = {'input': {'meta': meta_train_split
                      },
            'callback_input': {'meta_valid': meta_valid_split
                               }
            }

    sns_pipe = ship_no_ship_pipeline(config=CONFIG, train_mode=True)
    sns_pipe.fit_transform(data)


def train():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    meta_train = add_big_image_id(meta_train)
    meta_train_split, meta_valid_split = misc.train_test_split_with_empty_fraction_with_groups(meta_train,
                                                                                               groups=meta_train[
                                                                                                   ID_BIG_IMAGE],
                                                                                               empty_fraction=PARAMS.evaluation_empty_fraction,
                                                                                               test_size=PARAMS.evaluation_size,
                                                                                               shuffle=True,
                                                                                               random_state=SEED)

    meta_valid_split = meta_valid_split[meta_valid_split[IS_NOT_EMPTY_COLUMN] == 1].sample(
        PARAMS.in_train_evaluation_size, random_state=SEED)

    if DEV_MODE:
        meta_train_split = meta_train_split.sample(PARAMS.dev_mode_size, random_state=SEED)
        meta_valid_split = meta_valid_split.sample(int(PARAMS.dev_mode_size / 2), random_state=SEED)

    data = {'input': {'meta': meta_train_split
                      },
            'callback_input': {'meta_valid': meta_valid_split
                               }
            }

    pipeline = train_segmentation_pipeline(config=CONFIG)
    pipeline.fit_transform(data)


def evaluate():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    meta_train = add_big_image_id(meta_train)
    _, meta_valid_split = misc.train_test_split_with_empty_fraction_with_groups(meta_train,
                                                                                groups=meta_train[ID_BIG_IMAGE],
                                                                                empty_fraction=PARAMS.evaluation_empty_fraction,
                                                                                test_size=PARAMS.evaluation_size,
                                                                                shuffle=True, random_state=SEED)

    if DEV_MODE:
        _, meta_valid_split = misc.train_test_split_with_empty_fraction_with_groups(meta_valid_split,
                                                                                    groups=meta_valid_split[
                                                                                        ID_BIG_IMAGE],
                                                                                    empty_fraction=PARAMS.evaluation_empty_fraction,
                                                                                    test_size=PARAMS.evaluation_size,
                                                                                    shuffle=True, random_state=SEED)

    segm_pipe = inference_segmentation_pipeline(config=CONFIG)
    valid_ids = meta_valid_split[ID_COLUMN] + '.jpg'

    if INFERENCE_WITH_SHIP_NO_SHIP:
        sns_pipe = ship_no_ship_pipeline(config=CONFIG, train_mode=False)
        ids_ship, ids_no_ship = predict_ship_no_ship(meta_valid_split, sns_pipe, CHUNK_SIZE_SHIP_NO_SHIP)
        meta_valid_ship = meta_valid_split[valid_ids.isin(ids_ship)]
        prediction_ship = generate_submission(meta_valid_ship, segm_pipe, CHUNK_SIZE_SEGMENTATION)
        prediction = misc.combine_two_stage_predictions(ids_no_ship, prediction_ship, valid_ids)
    else:
        prediction = generate_submission(meta_valid_split, segm_pipe, CHUNK_SIZE_SEGMENTATION)

    gt = io.read_gt_subset(PARAMS.annotation_file, valid_ids)
    f2_per_image, image_ids = metrics.f_beta_metric(gt, prediction, beta=2, apply_mean=False)
    f2 = np.mean(f2_per_image)
    LOGGER.info('f2 {}'.format(f2))
    CTX.channel_send('f2', 0, f2)

    LOGGER.info('preparing results')
    results = misc.prepare_results(gt, prediction, meta_valid_split, f2_per_image, image_ids)
    results_filepath = os.path.join(EXPERIMENT_DIR, 'validation_results.csv')
    results.to_csv(results_filepath, index=None)


def predict():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_test = meta[meta['is_train'] == 0]

    if DEV_MODE:
        meta_test = meta_test.sample(PARAMS.dev_mode_size, random_state=SEED)

    segm_pipe = inference_segmentation_pipeline(config=CONFIG)
    test_ids = meta_test[ID_COLUMN] + '.jpg'

    if INFERENCE_WITH_SHIP_NO_SHIP:
        sns_pipe = ship_no_ship_pipeline(config=CONFIG, train_mode=False)
        ids_ship, ids_no_ship = predict_ship_no_ship(meta_test, sns_pipe, CHUNK_SIZE_SHIP_NO_SHIP)
        meta_test_ship = meta_test[test_ids.isin(ids_ship)]
        prediction_ship = generate_submission(meta_test_ship, segm_pipe, CHUNK_SIZE_SEGMENTATION)
        submission = misc.combine_two_stage_predictions(ids_no_ship, prediction_ship, test_ids)
    else:
        submission = generate_submission(meta_test, segm_pipe, CHUNK_SIZE_SEGMENTATION)

    submission_filepath = os.path.join(EXPERIMENT_DIR, 'submission.csv')

    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    LOGGER.info('submission saved to {}'.format(submission_filepath))
    LOGGER.info('submission head \n\n{}'.format(submission.head()))


#   __    __  .___________. __   __          _______.
#  |  |  |  | |           ||  | |  |        /       |
#  |  |  |  | `---|  |----`|  | |  |       |   (----`
#  |  |  |  |     |  |     |  | |  |        \   \
#  |  `--'  |     |  |     |  | |  `----.----)   |
#   \______/      |__|     |__| |_______|_______/
#

def add_big_image_id(meta):
    big_image_ids = pd.read_csv('big-images-ids_v2.csv')
    meta['ImageId'] = meta[ID_COLUMN] + '.jpg'
    meta_joined = pd.merge(meta, big_image_ids, on='ImageId')
    return meta_joined


def generate_submission(meta_data, pipeline, chunk_size):
    if chunk_size is not None:
        return _generate_submission_in_chunks(meta_data, pipeline, chunk_size)
    else:
        return _generate_submission(meta_data, pipeline)


def _generate_submission(meta_data, pipeline):
    prediction = _generate_prediction(meta_data, pipeline)
    submission = misc.create_submission(meta_data[ID_COLUMN] + '.jpg', prediction)
    return submission


def _generate_submission_in_chunks(meta_data, pipeline, chunk_size):
    submissions = []
    for meta_chunk in misc.generate_data_frame_chunks(meta_data, chunk_size):
        prediction_chunk = _generate_prediction(meta_chunk, pipeline)
        submission_chunk = misc.create_submission(meta_chunk[ID_COLUMN] + '.jpg', prediction_chunk)
        submissions.append(submission_chunk)
    submission = pd.concat(submissions)
    return submission


def _generate_prediction(meta_data, pipeline):
    data = {'input': {'meta': meta_data,
                      },
            'callback_input': {'meta_valid': None
                               }
            }
    output = pipeline.transform(data)
    y_pred = output['labeled_images']
    return y_pred


def predict_ship_no_ship(meta_data, pipeline, chunk_size):
    if chunk_size is not None:
        return _predict_ship_no_ship_in_chunks(meta_data, pipeline, chunk_size)
    else:
        return _predict_ship_no_ship(meta_data, pipeline)


def _predict_ship_no_ship(meta_data, pipeline):
    prediction = _generate_prediction_ship_no_ship(meta_data, pipeline)
    ids_ship, ids_no_ship = misc.get_ship_no_ship_ids(meta_data[ID_COLUMN] + '.jpg', prediction)
    return ids_ship, ids_no_ship


def _predict_ship_no_ship_in_chunks(meta_data, pipeline, chunk_size):
    ids_ship, ids_no_ship = [], []
    for meta_chunk in misc.generate_data_frame_chunks(meta_data, chunk_size):
        prediction_chunk = _generate_prediction_ship_no_ship(meta_chunk, pipeline)
        ids_ship_chunk, ids_no_ship_chunk = misc.get_ship_no_ship_ids(meta_chunk[ID_COLUMN] + '.jpg', prediction_chunk)
        ids_ship.extend(ids_ship_chunk)
        ids_no_ship.extend(ids_no_ship_chunk)
    return ids_ship, ids_no_ship


def _generate_prediction_ship_no_ship(meta_data, pipeline):
    data = {'input': {'meta': meta_data,
                      },
            'callback_input': {'meta_valid': None
                               }
            }
    output = pipeline.transform(data)
    y_pred = output['classes']
    return y_pred


#  .___  ___.      ___       __  .__   __.
#  |   \/   |     /   \     |  | |  \ |  |
#  |  \  /  |    /  ^  \    |  | |   \|  |
#  |  |\/|  |   /  /_\  \   |  | |  . `  |
#  |  |  |  |  /  _____  \  |  | |  |\   |
#  |__|  |__| /__/     \__\ |__| |__| \__|
#

if __name__ == '__main__':
    train_ship_no_ship()
    train()
    evaluate()
    predict()
