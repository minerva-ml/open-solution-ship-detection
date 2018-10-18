import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from toolkit.pytorch_transformers.models import Model
from torch.autograd import Variable

from common_blocks.architectures.classification import  Densenet
from common_blocks.utils.misc import get_list_of_image_predictions, sigmoid, softmax
from .architectures import encoders, unet, large_kernel_matters, pspnet
from . import callbacks as cbk
from .lovasz_losses import lovasz_hinge

ENCODERS = {'ResNet': {'model': encoders.ResNetEncoders,
                       'model_config': {'encoder_depth': 34, 'pretrained': True, 'pool0': True
                                        }
                       },
            'SeResNet': {'model': encoders.SeResNetEncoders,
                         'model_config': {'encoder_depth': 50, 'pretrained': 'imagenet', 'pool0': True
                                          }
                         },
            'SeResNetXt': {'model': encoders.SeResNetXtEncoders,
                           'model_config': {'encoder_depth': 101, 'pretrained': 'imagenet', 'pool0': True
                                            }
                           },
            'DenseNet': {'model': encoders.DenseNetEncoders,
                         'model_config': {'encoder_depth': 201, 'pretrained': 'imagenet', 'pool0': True
                                          }
                         },
            }

ARCHITECTURES = {'UNet': {'model': unet.UNet,
                          'model_config': {'use_hypercolumn': False, 'dropout_2d': 0.0, 'pool0': True
                                           }},
                 'LargeKernelMatters': {'model': large_kernel_matters.LargeKernelMatters,
                                        'model_config': {'kernel_size': 9, 'internal_channels': 21,
                                                         'dropout_2d': 0.0, 'use_relu': True, 'pool0': True
                                                         },
                                        },
                 'PSPNet': {'model': pspnet.PSPNet,
                            'model_config': {'use_hypercolumn': False, 'pool0': True
                                             },
                            },
                 }

SNS_ARCHITECTURES = {
    "Densenet": {'model': Densenet,
                 'model_config': {'pretrained': 'imagenet'}
                 }}


class SegmentationModel(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.activation_func = self.architecture_config['model_params']['activation']
        self.set_model()
        self.set_loss()
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.callbacks = callbacks_network(self.callbacks_config)

    def fit(self, datagen, validation_datagen=None, meta_valid=None):
        self._initialize_model_weights()

        if not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.callbacks.set_params(self, validation_datagen=validation_datagen, meta_valid=meta_valid)
        self.callbacks.on_train_begin()

        batch_gen, steps = datagen
        for epoch_id in range(self.training_config['epochs']):
            self.callbacks.on_epoch_begin()
            for batch_id, data in enumerate(batch_gen):
                self.callbacks.on_batch_begin()
                self.freeze_weights()
                metrics = self._fit_loop(data)
                self.callbacks.on_batch_end(metrics=metrics)
                if batch_id == steps:
                    break
            self.callbacks.on_epoch_end()
            if self.callbacks.training_break():
                break
        self.callbacks.on_train_end()
        return self

    def _fit_loop(self, data):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = Variable(X).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor).cuda())
        else:
            X = Variable(X)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor))

        self.optimizer.zero_grad()
        outputs_batch = self.model(X)
        partial_batch_losses = {}

        if len(self.output_names) == 1:
            for (name, loss_function, weight), target in zip(self.loss_function, targets_var):
                batch_loss = loss_function(outputs_batch, target) * weight
        else:
            for (name, loss_function, weight), output, target in zip(self.loss_function, outputs_batch, targets_var):
                partial_batch_losses[name] = loss_function(output, target) * weight
            batch_loss = sum(partial_batch_losses.values())
        partial_batch_losses['sum'] = batch_loss

        batch_loss.backward()
        self.optimizer.step()

        return partial_batch_losses

    def transform(self, datagen, validation_datagen=None, *args, **kwargs):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            if self.activation_func == 'softmax':
                outputs[name] = [softmax(single_prediction, axis=0) for single_prediction in prediction]
            elif self.activation_func == 'sigmoid':
                outputs[name] = [sigmoid(np.squeeze(mask)) for mask in prediction]
            else:
                raise Exception('Only softmax and sigmoid activations are allowed')
        return outputs

    def _transform(self, datagen, validation_datagen=None, **kwargs):
        self.model.eval()

        batch_gen, steps = datagen
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            if isinstance(data, (list, tuple)):
                X = data[0]
            else:
                X = data

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)
            outputs_batch = self.model(X)

            if len(self.output_names) == 1:
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                for name, output in zip(self.output_names, outputs_batch):
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
            if batch_id == steps:
                break
        self.model.train()
        outputs = {'{}_prediction'.format(name): get_list_of_image_predictions(outputs_) for name, outputs_ in
                   outputs.items()}
        return outputs

    def set_model(self):
        architecture_name = self.architecture_config['model_params']['architecture']
        encoder_name = self.architecture_config['model_params']['encoder']
        encoder = ENCODERS[encoder_name]
        architecture = ARCHITECTURES[architecture_name]

        self.model = architecture['model'](encoder=encoder['model'](**encoder['model_config']),
                                           num_classes=self.architecture_config['model_params']['out_channels'],
                                           **architecture['model_config'])
        self._initialize_model_weights = lambda: None

    def set_loss(self):
        if self.activation_func == 'softmax':
            raise NotImplementedError('No softmax loss defined')
        elif self.activation_func == 'sigmoid':

            loss_function = weighted_sum_loss
            # loss_function = nn.BCEWithLogitsLoss()
            # loss_function = DiceWithLogitsLoss()
            # loss_function = lovasz_loss
            # loss_function = FocalWithLogitsLoss()
        else:
            raise Exception('Only softmax and sigmoid activations are allowed')
        self.loss_function = [('mask', loss_function, 1.0)]

    def freeze_weights(self):
        # # freeze encoder
        # if isinstance(self.model, nn.DataParallel):
        #     encoder_params = self.model.module.encoder.parameters()
        # else:
        #     encoder_params = self.model.encoder.parameters()
        #
        # for parameter in encoder_params:
        #     parameter.requires_grad = False
        #
        # # freeze batchnorm
        # for m in self.model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False
        pass

    def load(self, filepath):
        self.model.eval()

        if not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model.cpu()
            self.model.load_state_dict(torch.load(filepath))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        return self


class BinaryModel(SegmentationModel):
    def __init__(self, architecture_config, training_config, callbacks_config, **kwargs):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.weight_regularization = weight_regularization
        self.set_model()
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])

        self.epochs = 10
        self.callbacks_config = callbacks_config
        self.callbacks = callbacks_ship_no_ship(self.callbacks_config)
        self.activation_func = 'sigmoid'
        self.validation_loss = {}

    def set_model(self):
        architecture = self.architecture_config['model_params']['architecture']
        config = SNS_ARCHITECTURES[architecture]
        self.model = config['model'](**config['model_config'])
        self._initialize_model_weights = lambda: None

    def set_loss(self):
        self.loss_function = [('ship_no_ship', nn.CrossEntropyLoss(), 1.0)]

    def freeze_weights(self):
        pass


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, reduction='elementwise_mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target):
        if not (target.size() == output.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(target.size(), output.size()))

        max_val = (-output).clamp(min=0)
        logpt = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()
        pt = torch.exp(-logpt)
        at = self.alpha * target + (1 - target)
        loss = at * ((1 - pt).pow(self.gamma)) * logpt

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'elementwise_mean':
            return loss.mean()
        else:
            return loss.sum()


class DiceWithLogitsLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        output = F.sigmoid(output)
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


def weighted_sum_loss(output, target):
    bce = nn.BCEWithLogitsLoss()(output, target)
    dice = DiceWithLogitsLoss()(output, target)
    return bce + 0.25 * dice


def lovasz_loss(output, target):
    target = target.long()
    return lovasz_hinge(output, target)


def weight_regularization(model, regularize, weight_decay_conv2d):
    if regularize:
        parameter_list = [
            {'params': filter(lambda p: p.requires_grad, model.parameters()),
             'weight_decay': weight_decay_conv2d},
        ]
    else:
        parameter_list = [filter(lambda p: p.requires_grad, model.parameters())]
    return parameter_list


def callbacks_network(callbacks_config):
    experiment_timing = cbk.ExperimentTiming(**callbacks_config['experiment_timing'])
    model_checkpoints = cbk.ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = cbk.ReduceLROnPlateauScheduler(**callbacks_config['reduce_lr_on_plateau_scheduler'])
    training_monitor = cbk.TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = cbk.ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = cbk.NeptuneMonitor(**callbacks_config['neptune_monitor'])
    early_stopping = cbk.EarlyStopping(**callbacks_config['early_stopping'])
    init_lr_finder = cbk.InitialLearningRateFinder()

    return cbk.CallbackList(
        callbacks=[experiment_timing, training_monitor, validation_monitor,
                   model_checkpoints, neptune_monitor, early_stopping,
                   lr_scheduler,  # init_lr_finder,
                   ])


def callbacks_ship_no_ship(callbacks_config):
    training_monitor = cbk.TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = cbk.SNS_ValidationMonitor()
    model_checkpoints = cbk.ModelCheckpoint(**callbacks_config['model_checkpoint'])
    one_cycle_callback = cbk.OneCycleCallback(**callbacks_config['one_cycle_scheduler'])

    return cbk.CallbackList([training_monitor, validation_monitor, model_checkpoints,
                             # one_cycle_callback
                             ])
