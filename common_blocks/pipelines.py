from functools import partial

from steppy.base import Step
from steppy.adapter import Adapter, E

from . import loaders



def preprocessing_train(config, model_name='network'):
    if config.general.loader_mode == 'resize':
        loader_config = config.loaders.resize
        LOADER = loaders.ImageSegmentationLoaderResize
    else:
        raise NotImplementedError

    reader_train = Step(name='xy_train',
                        transformer=loaders.MetaReader(train_mode=True, **config.meta_reader[model_name]),
                        input_data=['input'],
                        adapter=Adapter({'meta': E('input', 'meta')}))

    reader_inference = Step(name='xy_inference',
                            transformer=loaders.MetaReader(train_mode=True, **config.meta_reader[model_name]),
                            input_data=['callback_input'],
                            adapter=Adapter({'meta': E('callback_input', 'meta_valid')}))

    loader = Step(name='loader',
                  transformer=LOADER(train_mode=True, **loader_config),
                  input_steps=[reader_train, reader_inference],
                  adapter=Adapter({'X': E(reader_train.name, 'X'),
                                   'y': E(reader_train.name, 'y'),
                                   'X_valid': E(reader_inference.name, 'X'),
                                   'y_valid': E(reader_inference.name, 'y'),
                                   }))
    return loader


def preprocessing_inference(config, model_name='network'):
    if config.general.loader_mode == 'resize':
        loader_config = config.loaders.resize
        LOADER = loaders.ImageSegmentationLoaderResize
    else:
        raise NotImplementedError

    reader_inference = Step(name='xy_inference',
                            transformer=loaders.MetaReader(train_mode=False, **config.meta_reader[model_name]),
                            input_data=['input'],
                            adapter=Adapter({'meta': E('input', 'meta')}))

    loader = Step(name='loader',
                  transformer=LOADER(train_mode=False, **loader_config),
                  input_steps=[reader_inference],
                  adapter=Adapter({'X': E(reader_inference.name, 'X'),
                                   'y': E(reader_inference.name, 'y'),
                                   }))
    return loader


def preprocessing_inference_tta(config, model_name='network'):
    if config.general.loader_mode == 'resize':
        loader_config = config.loaders.resize_tta
        LOADER = loaders.ImageSegmentationLoaderResizeTTA
    else:
        raise NotImplementedError

    reader_inference = Step(name='reader_inference',
                            transformer=loaders.MetaReader(train_mode=False, **config.meta_reader[model_name]),
                            input_data=['input'],
                            adapter=Adapter({'meta': E('input', 'meta')}))

    tta_generator = Step(name='tta_generator',
                         transformer=loaders.MetaTestTimeAugmentationGenerator(**config.tta_generator),
                         input_steps=[reader_inference],
                         adapter=Adapter({'X': E('reader_inference', 'X')})
                         )

    loader = Step(name='loader',
                  transformer=LOADER(**loader_config),
                  input_steps=[tta_generator],
                  adapter=Adapter({'X': E(tta_generator.name, 'X_tta'),
                                   'tta_params': E(tta_generator.name, 'tta_params'),
                                   })
                  )
    return loader, tta_generator


def aggregator(name, model, tta_generator, config):
    tta_aggregator = Step(name=name,
                          transformer=loaders.TestTimeAugmentationAggregator(**config),
                          input_steps=[model, tta_generator],
                          adapter=Adapter({'images': E(model.name, 'mask_prediction'),
                                           'tta_params': E(tta_generator.name, 'tta_params'),
                                           'img_ids': E(tta_generator.name, 'img_ids'),
                                           })
                          )
    return tta_aggregator