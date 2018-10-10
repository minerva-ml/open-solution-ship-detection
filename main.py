from functools import partial
import os
import shutil

from attrdict import AttrDict
import neptune
import pandas as pd
from sklearn.model_selection import train_test_split
from steppy.base import Step, IdentityOperation
from steppy.adapter import Adapter, E

from common_blocks import augmentation as aug
from common_blocks import metrics
from common_blocks import models
from common_blocks import pipelines
from common_blocks.utils import misc, io
from common_blocks import postprocessing

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
CLONE_EXPERIMENT_DIR_FROM = ''  # When running eval in the cloud specify this as for example /input/SAL-14/output/experiment
OVERWRITE_EXPERIMENT_DIR = False
DEV_MODE = False

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
CHUNK_SIZE = 500
SEED = 1234
ID_COLUMN = 'id'
IS_NOT_EMPTY_COLUMN = 'is_not_empty'
X_COLUMN = 'file_path_image'
Y_COLUMN = 'file_path_mask'

x_columns = [X_COLUMN]
y_columns = [Y_COLUMN, IS_NOT_EMPTY_COLUMN]

CONFIG = AttrDict({
    'execution': {'experiment_dir': EXPERIMENT_DIR,
                  'num_workers': PARAMS.num_workers,
                  },
    'general': {'img_H-W': (PARAMS.image_h, PARAMS.image_w),
                'loader_mode': PARAMS.loader_mode,
                'num_classes': 2,
                'original_size': (768, 768),
                },
    'meta_reader': {
        'network': {'x_columns': x_columns,
                    'y_columns': y_columns,
                    },
    },
    'loaders': {'resize': {'dataset_params': {'h': PARAMS.image_h,
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
        'network': {
            'architecture_config': {'model_params': {'in_channels': PARAMS.image_channels,
                                                     'out_channels': PARAMS.network_output_channels,
                                                     'architecture': PARAMS.architecture,
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
                'filepath': os.path.join(EXPERIMENT_DIR, 'checkpoints', 'network', 'best.torch'),
                'epoch_every': 1,
                'metric_name': PARAMS.validation_metric_name,
                'minimize': PARAMS.minimize_validation_metric},
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
    },
    'tta_generator': {'flip_ud': False,
                      'flip_lr': True,
                      'rotation': False,
                      'color_shift_runs': 0},
    'tta_aggregator': {'tta_inverse_transform': aug.test_time_augmentation_inverse_transform,
                       'method': PARAMS.tta_aggregation_method,
                       'nthreads': PARAMS.num_threads
                       },
    'thresholder': {'threshold_masks': PARAMS.threshold_masks,
                    },
})


#  .______    __  .______    _______  __       __  .__   __.  _______     _______.
#  |   _  \  |  | |   _  \  |   ____||  |     |  | |  \ |  | |   ____|   /       |
#  |  |_)  | |  | |  |_)  | |  |__   |  |     |  | |   \|  | |  |__     |   (----`
#  |   ___/  |  | |   ___/  |   __|  |  |     |  | |  . `  | |   __|     \   \
#  |  |      |  | |  |      |  |____ |  `----.|  | |  |\   | |  |____.----)   |
#  | _|      |__| | _|      |_______||_______||__| |__| \__| |_______|_______/
#


def network(config, suffix='', train_mode=True):
    if train_mode:
        preprocessing = pipelines.preprocessing_train(config, model_name='network', suffix=suffix)
    else:
        preprocessing = pipelines.preprocessing_inference(config, suffix=suffix)

    network = misc.FineTuneStep(name='network{}'.format(suffix),
                                transformer=models.SegmentationModel(**config.model['network']),
                                input_data=['callback_input'],
                                input_steps=[preprocessing],
                                adapter=Adapter({'datagen': E(preprocessing.name, 'datagen'),
                                                 'validation_datagen': E(preprocessing.name, 'validation_datagen'),
                                                 'meta_valid': E('callback_input', 'meta_valid'),
                                                 }),
                                is_fittable=True,
                                force_fitting=True,
                                fine_tuning=config.model.network.training_config.fine_tuning)

    if config.general.loader_mode == 'resize_and_pad':
        size_adjustment_function = partial(postprocessing.crop_image, target_size=config.general.original_size)
    elif config.general.loader_mode == 'resize' or config.general.loader_mode == 'stacking':
        size_adjustment_function = partial(postprocessing.resize_image, target_size=config.general.original_size)
    else:
        raise NotImplementedError

    if train_mode:
        network.set_mode_train()
        network.set_parameters_upstream({'experiment_directory': config.execution.experiment_dir,
                                         })
        return network
    else:
        mask_resize = Step(name='mask_resize{}'.format(suffix),
                           transformer=misc.make_apply_transformer(size_adjustment_function,
                                                                   output_name='resized_images',
                                                                   apply_on=['images']),
                           input_steps=[network],
                           adapter=Adapter({'images': E(network.name, 'mask_prediction'),
                                            }))

        binarizer = Step(name='binarizer{}'.format(suffix),
                         transformer=misc.make_apply_transformer(
                             partial(postprocessing.binarize, threshold=config.thresholder.threshold_masks),
                             output_name='binarized_images',
                             apply_on=['images']),
                         input_steps=[mask_resize],
                         adapter=Adapter({'images': E(mask_resize.name, 'resized_images'),
                                          }))

        labeler = Step(name='labeler{}'.format(suffix),
                       transformer=misc.make_apply_transformer(postprocessing.label,
                                                               output_name='labeled_images',
                                                               apply_on=['images']),
                       input_steps=[binarizer],
                       adapter=Adapter({'images': E(binarizer.name, 'binarized_images'),
                                        }))

        labeler.set_mode_inference()
        labeler.set_parameters_upstream({'experiment_directory': config.execution.experiment_dir,
                                         'is_fittable': False
                                         })
        network.is_fittable = True
        return labeler


def network_tta(config, suffix=''):
    preprocessing, tta_generator = pipelines.preprocessing_inference_tta(config, model_name='network')

    network = Step(name='network{}'.format(suffix),
                   transformer=models.SegmentationModel(**config.model['network']),
                   input_data=['callback_input'],
                   input_steps=[preprocessing])

    tta_aggregator = pipelines.aggregator('tta_aggregator{}'.format(suffix), network,
                                          tta_generator=tta_generator,
                                          config=config.tta_aggregator)

    prediction_renamed = Step(name='prediction_renamed{}'.format(suffix),
                              transformer=IdentityOperation(),
                              input_steps=[tta_aggregator],
                              adapter=Adapter({'mask_prediction': E(tta_aggregator.name, 'aggregated_prediction')
                                               }))

    if config.general.loader_mode == 'resize_and_pad':
        size_adjustment_function = partial(postprocessing.crop_image, target_size=config.general.original_size)
    elif config.general.loader_mode == 'resize' or config.general.loader_mode == 'stacking':
        size_adjustment_function = partial(postprocessing.resize_image, target_size=config.general.original_size)
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize{}'.format(suffix),
                       transformer=misc.make_apply_transformer(size_adjustment_function,
                                                               output_name='resized_images',
                                                               apply_on=['images']),
                       input_steps=[prediction_renamed],
                       adapter=Adapter({'images': E(prediction_renamed.name, 'mask_prediction'),
                                        }))

    binarizer = Step(name='binarizer{}'.format(suffix),
                     transformer=misc.make_apply_transformer(
                         partial(postprocessing.binarize, threshold=config.thresholder.threshold_masks),
                         output_name='binarized_images',
                         apply_on=['images']),
                     input_steps=[mask_resize],
                     adapter=Adapter({'images': E(mask_resize.name, 'resized_images'),
                                      }))

    labeler = Step(name='labeler{}'.format(suffix),
                   transformer=misc.make_apply_transformer(postprocessing.label,
                                                           output_name='labeled_images',
                                                           apply_on=['images']),
                   input_steps=[binarizer],
                   adapter=Adapter({'images': E(binarizer.name, 'binarized_images'),
                                    }))

    labeler.set_mode_inference()
    labeler.set_parameters_upstream({'experiment_directory': config.execution.experiment_dir,
                                     'is_fittable': False
                                     })
    network.is_fittable = True
    return labeler


#   __________   ___  _______   ______  __    __  .___________. __    ______   .__   __.
#  |   ____\  \ /  / |   ____| /      ||  |  |  | |           ||  |  /  __  \  |  \ |  |
#  |  |__   \  V  /  |  |__   |  ,----'|  |  |  | `---|  |----`|  | |  |  |  | |   \|  |
#  |   __|   >   <   |   __|  |  |     |  |  |  |     |  |     |  | |  |  |  | |  . `  |
#  |  |____ /  .  \  |  |____ |  `----.|  `--'  |     |  |     |  | |  `--'  | |  |\   |
#  |_______/__/ \__\ |_______| \______| \______/      |__|     |__|  \______/  |__| \__|
#


def train():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    meta_train_split, meta_valid_split = train_test_split(meta_train,
                                                          stratify=meta_train[IS_NOT_EMPTY_COLUMN],
                                                          test_size=PARAMS.evaluation_size,
                                                          shuffle=True, random_state=SEED)
    _, meta_valid_split = train_test_split(meta_valid_split, stratify=meta_valid_split[IS_NOT_EMPTY_COLUMN],
                                           test_size=PARAMS.in_train_evaluation_size,
                                           shuffle=True, random_state=SEED)

    if DEV_MODE:
        meta_train_split = meta_train_split.sample(PARAMS.dev_mode_size, random_state=SEED)
        meta_valid_split = meta_valid_split.sample(int(PARAMS.dev_mode_size / 2), random_state=SEED)

    data = {'input': {'meta': meta_train_split
                      },
            'callback_input': {'meta_valid': meta_valid_split
                               }
            }

    pipeline_network = network(config=CONFIG, train_mode=True)
    pipeline_network.fit_transform(data)


def evaluate():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]

    _, meta_valid_split = train_test_split(meta_train,
                                           stratify=meta_train[IS_NOT_EMPTY_COLUMN],
                                           test_size=PARAMS.evaluation_size,
                                           shuffle=True, random_state=SEED)

    if DEV_MODE:
        meta_valid_split = meta_valid_split.sample(PARAMS.dev_mode_size, random_state=SEED)

    pipeline = network(config=CONFIG, train_mode=False)

    prediction = generate_submission(meta_valid_split, pipeline, CHUNK_SIZE)
    gt = io.read_gt_subset(PARAMS.annotation_file, meta_valid_split[ID_COLUMN] + '.jpg')
    f2 = metrics.f_beta_metric(gt, prediction, beta=2)
    LOGGER.info('F2 {}'.format(f2))
    CTX.channel_send('F2', 0, f2)


def predict():
    meta = pd.read_csv(PARAMS.metadata_filepath)
    meta_test = meta[meta['is_train'] == 0]

    if DEV_MODE:
        meta_test = meta_test.sample(PARAMS.dev_mode_size, random_state=SEED)

    pipeline = network(config=CONFIG, train_mode=False)

    submission = generate_submission(meta_test, pipeline, CHUNK_SIZE)

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


#  .___  ___.      ___       __  .__   __.
#  |   \/   |     /   \     |  | |  \ |  |
#  |  \  /  |    /  ^  \    |  | |   \|  |
#  |  |\/|  |   /  /_\  \   |  | |  . `  |
#  |  |  |  |  /  _____  \  |  | |  |\   |
#  |__|  |__| /__/     \__\ |__| |__| \__|
#

if __name__ == '__main__':
    train()
    evaluate()
    predict()
