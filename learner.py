import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon import Trainer
import numpy as np
import matplotlib.pyplot as plt
import math
import copy


class Constant():
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, iteration):
        return self.lr


class TriangularSchedule():
    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        inc_fraction: fraction of iterations spent in increasing stage (float)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction

    def __call__(self, iteration):
        if iteration <= self.cycle_length * self.inc_fraction:
            unit_cycle = iteration * 1 / \
                (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / \
                (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (
            unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle


class CosineAnnealingSchedule():
    def __init__(self, min_lr, max_lr, cycle_length):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length

    def __call__(self, iteration):
        if iteration <= self.cycle_length:
            unit_cycle = (
                1 + math.cos(iteration * math.pi / self.cycle_length)) / 2
            adjusted_cycle = (
                unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
            return adjusted_cycle
        else:
            return self.min_lr


class CyclicalSchedule():
    def __init__(self, schedule_class, cycle_length, cycle_length_decay=1,
                 cycle_magnitude_decay=1, **kwargs):
        """
        schedule_class: class of schedule, expected to take `cycle_length` argument
        cycle_length: iterations used for initial cycle (int)
        cycle_length_decay: factor multiplied to cycle_length each cycle (float)
        cycle_magnitude_decay: factor multiplied learning rate magnitudes each cycle (float)
        kwargs: passed to the schedule_class
        """
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.kwargs = kwargs

    def __call__(self, iteration):
        cycle_idx = 0
        cycle_length = self.length
        idx = self.length
        while idx <= iteration:
            cycle_length = math.ceil(cycle_length * self.length_decay)
            cycle_idx += 1
            idx += cycle_length
        cycle_offset = iteration - idx + cycle_length

        schedule = self.schedule_class(
            cycle_length=cycle_length, **self.kwargs)
        return schedule(cycle_offset) * self.magnitude_decay**cycle_idx


class LinearWarmUp():
    def __init__(self, schedule, start_lr, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        start_lr: learning rate used at start of the warm-up (float)
        length: number of iterations used for the warm-up (int)
        """
        self.schedule = schedule
        self.start_lr = start_lr
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.finish_lr = copy.copy(schedule)(0)
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.length:
            return iteration * (self.finish_lr - self.start_lr) / (self.length) + self.start_lr
        else:
            return self.schedule(iteration - self.length)


class LinearCoolDown():
    def __init__(self, schedule, finish_lr, start_idx, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        finish_lr: learning rate used at end of the cool-down (float)
        start_idx: iteration to start the cool-down (int)
        length: number of iterations used for the cool-down (int)
        """
        self.schedule = schedule
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.start_lr = copy.copy(self.schedule)(start_idx)
        self.finish_lr = finish_lr
        self.start_idx = start_idx
        self.finish_idx = start_idx + length
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.start_idx:
            return self.schedule(iteration)
        elif iteration <= self.finish_idx:
            return (iteration - self.start_idx) * (self.finish_lr - self.start_lr) / (self.length) + self.start_lr
        else:
            return self.finish_lr


class OneCycleSchedule():
    def __init__(self, start_lr, max_lr, cycle_length, cooldown_length=0, finish_lr=None):
        """
        start_lr: lower bound for learning rate in triangular cycle (float)
        max_lr: upper bound for learning rate in triangular cycle (float)
        cycle_length: iterations between start and finish of triangular cycle: 2x 'stepsize' (int)
        cooldown_length: number of iterations used for the cool-down (int)
        finish_lr: learning rate used at end of the cool-down (float)
        """
        if (cooldown_length > 0) and (finish_lr is None):
            raise ValueError(
                "Must specify finish_lr when using cooldown_length > 0.")
        if (cooldown_length == 0) and (finish_lr is not None):
            raise ValueError(
                "Must specify cooldown_length > 0 when using finish_lr.")

        finish_lr = finish_lr if (cooldown_length > 0) else start_lr
        schedule = TriangularSchedule(
            min_lr=start_lr, max_lr=max_lr, cycle_length=cycle_length)
        self.schedule = LinearCoolDown(
            schedule, finish_lr=finish_lr, start_idx=cycle_length, length=cooldown_length)

    def __call__(self, iteration):
        return self.schedule(iteration)


class Learner:
    """docstring for Learner"""

    def __init__(self, net, criterion, optimizer, scheduler, loader, metrics, augmentor=None, ctx=mx.gpu(0)):
        self.net = net
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.loader = loader
        self.metrics = metrics
        self.augmentor = augmentor
        self.ctx = ctx

    def train(self, epochs, wd, params=None, init_epochs=0, bs=4):
        trainer = Trainer(self.net.collect_params(
            params), self.optimizer, {'wd': wd})
        metrics = mx.metric.create(self.metrics)
        self.history = [[], []]
        iteration = 1
        val_iter = 1
        avg_mom = 0.98
        tavg_loss, vavg_loss = 0., 0.

        for epoch in range(epochs):
            for data, label in self.loader[0]:
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                with autograd.record():
                    output = self.net(data)
                    loss = self.criterion(output, label)
                lr = self.scheduler(iteration)
                trainer.set_learning_rate(lr)
                loss.backward()
                trainer.step(bs)
                tavg_loss = tavg_loss * avg_mom + \
                    (1 - avg_mom) * (nd.mean(loss).asscalar())
                self.history[0].append(tavg_loss / (1 - avg_mom ** iteration))
                iteration += 1

            metrics.reset()

            for data, label in self.loader[1]:
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                output = self.net(data)
                loss = self.criterion(output, label)
                vavg_loss = vavg_loss * avg_mom + \
                    (1 - avg_mom) * (nd.mean(loss).asscalar())
                self.history[1].append(vavg_loss / (1 - avg_mom ** val_iter))
                val_iter += 1
                metrics.update(preds=output, labels=label)
            status = [init_epochs + epoch + 1] + \
                [self.history[0][-1], self.history[1][-1]]
            if self.metrics is not None:
                status.append(metrics.get()[1])
            print('{}'.format(status))
        return self.history

    def test(self, loader):
        metrics = mx.metric.create(self.metrics)
        tst_loss = []

        for data, label in loader:
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            output = self.net(data)
            loss = self.criterion(output, label)
            tst_loss.append(nd.mean(loss).asscalar())
            # metrics.update(predict=output, label=label)
        status = [sum(tst_loss) / len(tst_loss)]
        if self.metrics is not None:
            status.append(metrics.get()[1])
        print('{}'.format(status))

    def find_lr(self, wds, bs=4):
        def __find(wd=0, init_value=1e-8, final_value=40., beta=0.98):
            trainer = mx.gluon.Trainer(
                self.net.collect_params(), self.optimizer, {"wd": wd})
            num = len(self.loader[0]) - 1
            mult = (final_value / init_value) ** (1 / num)
            lr = init_value
            trainer.set_learning_rate(lr)
            avg_loss = 0.
            best_loss = 0.
            batch_num = 0
            losses = []
            lrs = []
            for inputs, labels in self.loader[0]:
                batch_num += 1
                # As before, get the loss for this mini-batch of inputs/outputs
                inputs = inputs.as_in_context(self.ctx)
                labels = labels.as_in_context(self.ctx)
                with autograd.record():
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                # Compute the smoothed loss
                avg_loss = beta * avg_loss + \
                    (1 - beta) * nd.mean(loss).asscalar()
                smoothed_loss = avg_loss / (1 - beta**batch_num)
                # Stop if the loss is exploding
                if batch_num > 1 and smoothed_loss > 4 * best_loss:
                    self.net.load_parameters('tmp/params.net')
                    return lrs, losses
                # Record the best loss
                if smoothed_loss < best_loss or batch_num == 1:
                    best_loss = smoothed_loss
                # Store the values
                losses.append(smoothed_loss)
                lrs.append(lr)
                # Do the SGD step
                loss.backward()
                trainer.step(bs)
                # Update the lr for the next step
                lr *= mult
                trainer.set_learning_rate(lr)
            self.net.load_parameters('tmp/params.net')
            return lrs, losses
        result = []
        for wd in wds:
            lr, loss = __find(wd)
            result.append((lr, loss))
            plt.plot(lr[10:-5], loss[10:-5], label=str(wd))
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.legend()
        return result


class MeanIOU(mx.metric.EvalMetric):
    def __init__(self, classes, axis=1, name="mIOU", output_names=None, label_names=None):
        super(MeanIOU, self).__init__(name, axis=axis,
                                      output_names=output_names, label_names=label_names)
        self.axis = axis
        self.name = name
        self.classes = classes

    def update(self, labels, preds):
        # labels, preds = mx.metric.check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            pred_label = nd.argmax(pred_label, axis=self.axis - 1)
            pred_label = pred_label.asnumpy()
            label = label.asnumpy()
            k = [np.logical_and(pred_label == c, label == c).sum() /
                 np.logical_or(pred_label == c, label == c).sum() for c in range(self.classes)]
            self.sum_metric += sum(k) / self.classes
            self.num_inst += 1


class Dice(mx.metric.EvalMetric):
    def __init__(self, classes, axis=1, name="dice", output_names=None, label_names=None):
        super(Dice, self).__init__(name, axis=axis,
                                   output_names=output_names, label_names=label_names)
        self.axis = axis
        self.name = name
        self.classes = classes

    def update(self, labels, preds):
        # labels, preds = mx.metric.check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            pred_label = nd.argmax(pred_label, axis=self.axis - 1)
            pred_label = pred_label.asnumpy()
            label = label.asnumpy()
            k = [np.logical_and(pred_label == c, label == c).sum(
            ) / np.logical_or(pred_label == c, label == c).sum() for c in range(self.classes)]

            J = sum(k) / self.classes
            self.sum_metric += (2 * J) / (1 + J)
            self.num_inst += 1
