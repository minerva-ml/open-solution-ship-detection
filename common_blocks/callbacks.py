import os
from datetime import datetime, timedelta
from functools import partial
from tempfile import TemporaryDirectory

import neptune
import numpy as np
import torch
from PIL import Image
from steppy.adapter import Adapter, E
from steppy.base import Step
from toolkit.pytorch_transformers.utils import Averager, persist_torch_model
from toolkit.pytorch_transformers.validation import score_model
from torch.autograd import Variable
from torch.autograd import Variable as V
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from common_blocks.utils.io import read_masks
from common_blocks.utils.misc import OneCycle, get_list_of_image_predictions, get_logger, make_apply_transformer, \
    sigmoid, softmax
from .metrics import intersection_over_union_thresholds
from .postprocessing import binarize, label, resize_image

logger = get_logger()

Y_COLUMN = 'file_path_mask'
ORIGINAL_SIZE = (768, 768)
THRESHOLD = 0.5
NUM_THREADS = 300


class Callback:
    def __init__(self):
        self.epoch_id = None
        self.batch_id = None

        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.output_names = None
        self.validation_datagen = None
        self.lr_scheduler = None

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen
        self.transformer = transformer

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0

    def on_train_end(self, *args, **kwargs):
        pass

    def on_epoch_begin(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        self.epoch_id += 1

    def training_break(self, *args, **kwargs):
        return False

    def on_batch_begin(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        self.batch_id += 1

    def get_validation_loss(self):
        if self.epoch_id not in self.transformer.validation_loss.keys():
            self.transformer.validation_loss[self.epoch_id] = score_model(self.model, self.loss_function,
                                                                          self.validation_datagen)

        return self.transformer.validation_loss[self.epoch_id]


class CallbackList:
    def __init__(self, callbacks=None):
        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = callbacks

    def __len__(self):
        return len(self.callbacks)

    def set_params(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.set_params(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(*args, **kwargs)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def training_break(self, *args, **kwargs):
        callback_out = [callback.training_break(*args, **kwargs) for callback in self.callbacks]
        return any(callback_out)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(*args, **kwargs)


class TrainingMonitor(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        self.epoch_loss_averagers = {}
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averagers = {}
        self.epoch_id = 0
        self.batch_id = 0

    def on_epoch_end(self, *args, **kwargs):
        for name, averager in self.epoch_loss_averagers.items():
            epoch_avg_loss = averager.value
            averager.reset()
            if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
                logger.info('epoch {0} {1}:     {2:.5f}'.format(self.epoch_id, name, epoch_avg_loss))
        self.epoch_id += 1

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]

            if name in self.epoch_loss_averagers.keys():
                self.epoch_loss_averagers[name].send(loss)
            else:
                self.epoch_loss_averagers[name] = Averager()
                self.epoch_loss_averagers[name].send(loss)

            if self.batch_every and ((self.batch_id % self.batch_every) == 0):
                logger.info('epoch {0} batch {1} {2}:     {3:.5f}'.format(self.epoch_id, self.batch_id, name, loss))
        self.batch_id += 1


class ExponentialLRScheduler(Callback):
    def __init__(self, gamma, epoch_every=1, batch_every=None):
        super().__init__()
        self.gamma = gamma
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.lr_scheduler = ExponentialLR(self.optimizer, self.gamma, last_epoch=-1)

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('initial lr: {0}'.format(self.optimizer.state_dict()['param_groups'][0]['initial_lr']))

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and (((self.epoch_id + 1) % self.epoch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} current lr: {1}'.format(self.epoch_id + 1,
                                                           self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.epoch_id += 1

    def on_batch_end(self, *args, **kwargs):
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} batch {1} current lr: {2}'.format(
                self.epoch_id + 1, self.batch_id + 1, self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.batch_id += 1


class ReduceLROnPlateauScheduler(Callback):
    def __init__(self, metric_name, minimize, reduce_factor, reduce_patience, min_lr):
        super().__init__()
        self.ctx = neptune.Context()
        self.ctx.channel_reset('Learning Rate')
        self.metric_name = metric_name
        self.minimize = minimize
        self.reduce_factor = reduce_factor
        self.reduce_patience = reduce_patience
        self.min_lr = min_lr

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        super().set_params(transformer, validation_datagen)
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer,
                                              mode='min' if self.minimize else 'max',
                                              factor=self.reduce_factor,
                                              patience=self.reduce_patience,
                                              min_lr=self.min_lr)

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0

    def on_epoch_end(self, *args, **kwargs):
        self.model.eval()
        val_loss = self.get_validation_loss()
        metric = val_loss[self.metric_name]
        metric = metric.data.cpu().numpy()[0]
        self.model.train()

        self.lr_scheduler.step(metrics=metric, epoch=self.epoch_id)
        logger.info('epoch {0} current lr: {1}'.format(self.epoch_id + 1,
                                                       self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.ctx.channel_send('Learning Rate', x=self.epoch_id,
                              y=self.optimizer.state_dict()['param_groups'][0]['lr'])

        self.epoch_id += 1


class InitialLearningRateFinder(Callback):
    def __init__(self, min_lr=1e-8, multipy_factor=1.05, add_factor=0.0):
        super().__init__()
        self.ctx = neptune.Context()
        self.ctx.channel_reset('Learning Rate Finder')
        self.min_lr = min_lr
        self.multipy_factor = multipy_factor
        self.add_factor = add_factor

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        super().set_params(transformer, validation_datagen)
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]
        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        logger.info('Learning Rate {} Loss {})'.format(current_lr, loss))
        self.ctx.channel_send('Learning Rate Finder', x=self.batch_id, y=current_lr)
        self.ctx.channel_send('Loss', x=self.batch_id, y=loss)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr * self.multipy_factor + self.add_factor
        self.batch_id += 1


class ExperimentTiming(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every
        self.batch_start = None
        self.epoch_start = None
        self.current_sum = None
        self.current_mean = None

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('starting training...')

    def on_train_end(self, *args, **kwargs):
        logger.info('training finished')

    def on_epoch_begin(self, *args, **kwargs):
        if self.epoch_id > 0:
            epoch_time = datetime.now() - self.epoch_start
            if self.epoch_every:
                if (self.epoch_id % self.epoch_every) == 0:
                    logger.info('epoch {0} time {1}'.format(self.epoch_id - 1, str(epoch_time)[:-7]))
        self.epoch_start = datetime.now()
        self.current_sum = timedelta()
        self.current_mean = timedelta()
        logger.info('epoch {0} ...'.format(self.epoch_id))

    def on_batch_begin(self, *args, **kwargs):
        if self.batch_id > 0:
            current_delta = datetime.now() - self.batch_start
            self.current_sum += current_delta
            self.current_mean = self.current_sum / self.batch_id
        if self.batch_every:
            if self.batch_id > 0 and (((self.batch_id - 1) % self.batch_every) == 0):
                logger.info('epoch {0} average batch time: {1}'.format(self.epoch_id, str(self.current_mean)[:-5]))
        if self.batch_every:
            if self.batch_id == 0 or self.batch_id % self.batch_every == 0:
                logger.info('epoch {0} batch {1} ...'.format(self.epoch_id, self.batch_id))
        self.batch_start = datetime.now()


class NeptuneMonitor(Callback):
    def __init__(self, image_nr, image_resize, image_every, model_name):
        super().__init__()
        self.model_name = model_name
        self.ctx = neptune.Context()
        self.epoch_loss_averager = Averager()
        self.image_resize = image_resize
        self.image_every = image_every
        self.image_nr = image_nr

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averagers = {}
        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]

            if name in self.epoch_loss_averagers.keys():
                self.epoch_loss_averagers[name].send(loss)
            else:
                self.epoch_loss_averagers[name] = Averager()
                self.epoch_loss_averagers[name].send(loss)

            self.ctx.channel_send(name, loss)
        self.batch_id += 1

    def on_epoch_end(self, *args, **kwargs):
        self._send_numeric_channels()
        if self.image_every is not None and self.epoch_id % self.image_every == 0:
            self._send_image_channels()
        self.epoch_id += 1

    def _send_numeric_channels(self, *args, **kwargs):
        for name, averager in self.epoch_loss_averagers.items():
            epoch_avg_loss = averager.value
            averager.reset()
            self.ctx.channel_send('{} epoch {} loss'.format(self.model_name, name), x=self.epoch_id, y=epoch_avg_loss)

        self.model.eval()
        val_loss = self.get_validation_loss()
        self.model.train()
        for name, loss in val_loss.items():
            loss = loss.data.cpu().numpy()[0]
            self.ctx.channel_send('{} epoch_val {} loss'.format(self.model_name, name), x=self.epoch_id, y=loss)

    def _send_image_channels(self):
        self.model.eval()
        image_triplets = self._get_image_triplets()
        if self.image_nr is not None:
            image_triplets = image_triplets[:self.image_nr]
        self.model.train()

        for i, (raw, pred, truth) in enumerate(image_triplets):
            h, w, _ = raw.shape
            image_glued = np.zeros((h, 3 * w + 20, 3))
            image_glued[:, :w, :] = raw
            image_glued[:, (w + 10):(2 * w + 10), :] = pred
            image_glued[:, (2 * w + 20):, :] = truth

            pill_image = Image.fromarray((image_glued * 255.).astype(np.uint8))
            h_, w_, _ = image_glued.shape
            pill_image = pill_image.resize((int(self.image_resize * w_), int(self.image_resize * h_)),
                                           Image.ANTIALIAS)

            self.ctx.channel_send('{} predictions'.format(self.model_name), neptune.Image(
                name='epoch{}_batch{}_idx{}'.format(self.epoch_id, self.batch_id, i),
                description="image, prediction, ground truth",
                data=pill_image))

    def _get_image_triplets(self):
        image_triplets = []
        batch_gen, steps = self.validation_datagen
        for batch_id, data in enumerate(batch_gen):
            predictions, targets_tensors = self._get_predictions_targets(data)

            raw_images = data[0].numpy()
            ground_truth_masks = targets_tensors[0].cpu().numpy()

            for image, prediction, target in zip(raw_images, predictions, ground_truth_masks):
                raw = denormalize(image).transpose(1, 2, 0)
                pred = np.tile(prediction[1, :, :], (3, 1, 1)).transpose(1, 2, 0)
                truth = np.tile(target[1, :, :], (3, 1, 1)).transpose(1, 2, 0)
                image_triplets.append((raw, pred, truth))
            break
        return image_triplets

    def _get_predictions_targets(self, data):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = Variable(X, volatile=True).cuda()
        else:
            X = Variable(X, volatile=True)

        predictions = sigmoid(self.model(X).data.cpu().numpy())
        return predictions, targets_tensors


class SNS_ValidationMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.ctx = neptune.Context()
        self.best_loss = None

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        self.transformer = transformer
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.loss_function = transformer.loss_function

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]
            self.ctx.channel_send(name, loss)

    def on_epoch_end(self, *args, **kwargs):
        self.model.eval()
        self.epoch_id += 1
        self.validation_loss = self.calculate_epoch_end_metrics()
        epoch_end_loss = self.validation_loss['sum']
        epoch_end_acc = self.validation_loss['acc']

        self.transformer.validation_loss[self.epoch_id] = {'acc': V(torch.Tensor([epoch_end_acc])),
                                                           'sum': V(torch.Tensor([float(epoch_end_loss)]))}
        logger.info('epoch {0} ship no ship epoch end validation loss: {1}'.format(self.epoch_id, epoch_end_loss))
        logger.info('epoch {0} ship no ship epoch end accuracy: {1}'.format(self.epoch_id, epoch_end_acc))
        self.ctx.channel_send("ship_no_ship_epoch_end_acc", epoch_end_acc)
        self.ctx.channel_send("ship_no_ship_epoch_end_loss", epoch_end_loss)

    def calculate_epoch_end_metrics(self):
        self.model.eval()
        batch_gen, steps = self.validation_datagen
        sum_loss = 0
        Ys = []
        Ypreds = []
        for batch in batch_gen:
            X, y = batch
            X, y = V(X, volatile=True), V(y, volatile=True)
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()

            y_pred = self.model(X)
            loss_val = self.loss_function[0][1](y_pred, y.long())
            y_pred = y_pred.max(1)[1]
            y_pred = y_pred.data.cpu().numpy().astype(int).ravel()
            y = y.data.cpu().numpy()
            Ys.append(y)
            Ypreds.append(y_pred)
            sum_loss += loss_val.data.cpu().numpy()

        Ys = np.concatenate(Ys)
        Ypreds = (np.concatenate(Ypreds) > 0.5).astype(int)
        matches = sum(Ys == Ypreds)
        acc = matches / float(len(Ys))

        self.model.train()

        return {'sum': sum_loss / steps, "acc": acc}

    def get_validation_loss(self):
        if not self.transformer.validation_loss:
            self.transformer.validation_loss = {}
        return self.transformer.validation_loss[self.epoch_id]


class ValidationMonitor(Callback):
    def __init__(self, data_dir, loader_mode, epoch_every=None, batch_every=None):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

        self.data_dir = data_dir
        self.validation_pipeline = postprocessing_pipeline_simplified
        self.loader_mode = loader_mode
        self.meta_valid = None
        self.y_true = None
        self.activation_func = None

    def set_params(self, transformer, validation_datagen, meta_valid=None, *args, **kwargs):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen
        self.meta_valid = meta_valid
        self.y_true = read_masks(self.meta_valid[Y_COLUMN].values)
        self.activation_func = transformer.activation_func
        self.transformer = transformer

    def get_validation_loss(self):
        return self._get_validation_loss()

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = self.get_validation_loss()
            self.model.train()
            for name, loss in val_loss.items():
                loss = loss.data.cpu().numpy()[0]
                logger.info('epoch {0} validation {1}:     {2:.5f}'.format(self.epoch_id, name, loss))
        self.epoch_id += 1

    def _get_validation_loss(self):
        output, epoch_loss = self._transform()

        logger.info('Calculating F2 Score')
        y_pred = self._generate_prediction(output)
        f2_score = intersection_over_union_thresholds(self.y_true, y_pred)
        logger.info('F2 score on validation is {}'.format(f2_score))

        if not self.transformer.validation_loss:
            self.transformer.validation_loss = {}
        self.transformer.validation_loss.setdefault(self.epoch_id, {'sum': epoch_loss,
                                                                    'f2': Variable(torch.Tensor([f2_score])),
                                                                    })
        return self.transformer.validation_loss[self.epoch_id]

    def _transform(self):
        self.model.eval()
        batch_gen, steps = self.validation_datagen
        partial_batch_losses = []
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            targets_var, outputs_batch = self._get_targets_and_output(data)

            if len(self.output_names) == 1:
                for (name, loss_function_one, weight), target in zip(self.loss_function, targets_var):
                    loss_sum = loss_function_one(outputs_batch, target) * weight
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                batch_losses = []
                for (name, loss_function_one, weight), output, target in zip(self.loss_function, outputs_batch,
                                                                             targets_var):
                    loss = loss_function_one(output, target) * weight
                    batch_losses.append(loss)
                    partial_batch_losses.setdefault(name, []).append(loss)
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
                loss_sum = sum(batch_losses)
            partial_batch_losses.append(loss_sum)
            if batch_id == steps:
                break
        self.model.train()
        average_losses = sum(partial_batch_losses) / steps
        outputs = {'{}_prediction'.format(name): get_list_of_image_predictions(outputs_) for name, outputs_ in
                   outputs.items()}
        for name, prediction in outputs.items():
            if self.activation_func == 'softmax':
                outputs[name] = [softmax(single_prediction, axis=0) for single_prediction in prediction]
            elif self.activation_func == 'sigmoid':
                outputs[name] = [sigmoid(np.squeeze(mask)) for mask in prediction]
            else:
                raise Exception('Only softmax and sigmoid activations are allowed')

        return outputs, average_losses

    def _get_targets_and_output(self, data):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = Variable(X, volatile=True).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor, volatile=True).cuda())
        else:
            X = Variable(X, volatile=True)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor, volatile=True))
        outputs_batch = self.model(X)

        return targets_var, outputs_batch

    def _generate_prediction(self, outputs):
        data = {'callback_input': {'meta': self.meta_valid,
                                   'meta_valid': None,
                                   },
                'network_output': {**outputs}
                }
        with TemporaryDirectory() as cache_dirpath:
            pipeline = self.validation_pipeline(cache_dirpath, self.loader_mode)
            output = pipeline.transform(data)
        y_pred = output['labeled_images']
        return y_pred


class ModelCheckpoint(Callback):
    def __init__(self, filepath, metric_name='sum', epoch_every=1, minimize=True):
        self.filepath = filepath
        self.minimize = minimize
        self.best_score = None

        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every

        self.metric_name = metric_name

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = self.get_validation_loss()
            loss_sum = val_loss[self.metric_name]
            loss_sum = loss_sum.data.cpu().numpy()[0]

            self.model.train()

            if self.best_score is None:
                self.best_score = loss_sum

            if (self.minimize and loss_sum < self.best_score) or (not self.minimize and loss_sum > self.best_score) or (
                    self.epoch_id == 0):
                self.best_score = loss_sum
                persist_torch_model(self.model, self.filepath)
                logger.info('epoch {0} model saved to {1}'.format(self.epoch_id, self.filepath))

        self.epoch_id += 1


class OneCycleCallback(Callback):
    def __init__(self, number_of_batches_per_full_cycle, max_lr, enabled=1, momentum_range=(0.95, 0.8),
                 prcnt_annihilate=10,
                 div=10):
        super().__init__()

        self.enabled = enabled
        self.number_of_batches_per_full_cycle = number_of_batches_per_full_cycle
        self.max_lr = max_lr
        self.momentum_range = momentum_range
        self.prcnt_annihilate = prcnt_annihilate
        self.div = div
        self.ctx = neptune.Context()

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        super().set_params(transformer, validation_datagen)
        self.optimizer = transformer.optimizer
        self.onecycle = OneCycle(self.number_of_batches_per_full_cycle,
                                 max_lr=self.max_lr,
                                 optimizer=self.optimizer,
                                 prcnt=self.prcnt_annihilate,
                                 div=self.div
                                 )

    def on_batch_end(self, *args, **kwargs):
        if self.enabled:
            lr, mom = self.onecycle.batch_step()
            self.ctx.channel_send("lr", lr)
            self.ctx.channel_send("momentum", mom)


class EarlyStopping(Callback):
    def __init__(self, metric_name='sum', patience=1000, minimize=True):
        self.patience = patience
        self.minimize = minimize
        self.best_score = None
        self.epoch_since_best = 0
        self._training_break = False
        self.metric_name = metric_name

    def training_break(self, *args, **kwargs):
        return self._training_break

    def on_epoch_end(self, *args, **kwargs):
        self.model.eval()
        val_loss = self.get_validation_loss()
        loss_sum = val_loss[self.metric_name]
        loss_sum = loss_sum.data.cpu().numpy()[0]

        self.model.train()

        if not self.best_score:
            self.best_score = loss_sum

        if (self.minimize and loss_sum < self.best_score) or (not self.minimize and loss_sum > self.best_score):
            self.best_score = loss_sum
            self.epoch_since_best = 0
        else:
            self.epoch_since_best += 1

        if self.epoch_since_best > self.patience:
            self._training_break = True
        self.epoch_id += 1


def denormalize(x):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return x * std + mean


def postprocessing_pipeline_simplified(cache_dirpath, loader_mode):
    if loader_mode == 'resize':
        size_adjustment_function = partial(resize_image, target_size=ORIGINAL_SIZE)
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize',
                       transformer=make_apply_transformer(size_adjustment_function,
                                                          output_name='resized_images',
                                                          apply_on=['images'],
                                                          n_threads=NUM_THREADS),
                       input_data=['network_output'],
                       adapter=Adapter({'images': E('network_output', 'mask_prediction'),
                                        }))

    binarizer = Step(name='binarizer',
                     transformer=make_apply_transformer(
                         partial(binarize, threshold=THRESHOLD),
                         output_name='binarized_images',
                         apply_on=['images'],
                         n_threads=NUM_THREADS),
                     input_steps=[mask_resize],
                     adapter=Adapter({'images': E(mask_resize.name, 'resized_images'),
                                      }))

    labeler = Step(name='labeler',
                   transformer=make_apply_transformer(
                       label,
                       output_name='labeled_images',
                       apply_on=['images'],
                       n_threads=NUM_THREADS),
                   input_steps=[binarizer],
                   adapter=Adapter({'images': E(binarizer.name, 'binarized_images'),
                                    }))

    labeler.set_mode_inference()
    labeler.set_parameters_upstream({'experiment_directory': cache_dirpath,
                                     'is_fittable': False
                                     })
    return labeler
