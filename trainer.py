import torch
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
from attack import FastGradientSignUntargeted
from metrics import RetrivalAccMetric


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, model_save_dir,
        metrics=[], start_epoch=0, criterion=None, domain_cls=False, unsup_da=False, adv_train=False, adv_epsilon=None,
        adv_alph=None, adv_iter=None, eval_train_dataset=None, eval_test_dataset=None):

    best_val_loss = None
    start_time = time.time()

    # for epoch in range(0, start_epoch):
    #     scheduler.step()
    writer = SummaryWriter('runs/{}'.format(model_save_dir))

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        if adv_train:
            attack = FastGradientSignUntargeted(model, adv_epsilon, adv_alph, min_val=-2.5, max_val=3.0,
                                                max_iters=adv_iter, _type='linf')  # l2
        else:
            attack = None

        if not domain_cls:
            train_loss, metrics = train_epoch(train_loader, model, attack, loss_fn, optimizer, cuda, writer, epoch,
                                              log_interval, metrics, adv_train)
            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f} Elapsed time: {}s'\
                .format(epoch + 1, n_epochs, train_loss, int(time.time() - start_time))

        else:
            train_loss, train_loss_sim, train_loss_domain_cls, metrics = train_domain_classifier_epoch(train_loader,\
                    model, attack, loss_fn, criterion, optimizer, cuda, writer, epoch, log_interval, metrics, unsup_da)
            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f} loss-sim: {:.4f} loss-domain-cls: {:.4f} Elapsed time: {}s'\
                .format(epoch + 1, n_epochs, train_loss, train_loss_sim, train_loss_domain_cls, int(time.time() - start_time))

        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

        summ_step = (epoch + 1) * len(train_loader) - 1
        # Test stage
        if not domain_cls:
            val_loss, val_loss_sim, val_loss_mix_div, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
            val_loss /= len(val_loader)
            val_loss_sim /= len(val_loader)
            val_loss_mix_div /= len(val_loader)
            writer.add_scalars('Loss/total', {'validation': val_loss}, summ_step)
            writer.add_scalars('Loss/similarity', {'validation': val_loss_sim}, summ_step)
            writer.add_scalars('Loss/miture-divergence', {'validation': val_loss_mix_div}, summ_step)
            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f} loss-sim: {:.4f} loss-mixture-div: {:.4f}'\
                .format(epoch + 1, n_epochs, val_loss, val_loss_sim, val_loss_mix_div)

        else:
            val_loss, val_loss_sim, val_loss_domain_cls, metrics = test_domain_classifier_epoch(val_loader, model,
                                                                        loss_fn, criterion, cuda, metrics, unsup_da)
            val_loss /= len(val_loader)
            val_loss_sim /= len(val_loader)
            val_loss_domain_cls /= len(val_loader)
            writer.add_scalars('Loss/total', {'validation': val_loss}, summ_step)
            writer.add_scalars('Loss/similarity', {'validation': val_loss_sim}, summ_step)
            writer.add_scalars('Loss/domain_clf', {'validation': val_loss_domain_cls}, summ_step)
            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f} loss-sim: {:.4f} loss-domain-cls: {:.4f}'\
                .format(epoch + 1, n_epochs, val_loss, val_loss_sim, val_loss_domain_cls)

        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
        torch.save(model.module.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(str(epoch).zfill(5))))

        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            writer.add_scalars('Metric/{}'.format(metric.name()), {'validation': metric.value()}, summ_step)

        scheduler.step(val_loss)
        print(message)


def train_epoch(train_loader, model, attack, loss_fn, optimizer, cuda, writer, epoch, log_interval, metrics, adv_train):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = {'sim': [], 'mixture_div': []}
    total_loss = 0
    total_loss_sim = 0
    total_loss_mix_div = 0
    num_iter = epoch * len(train_loader)

    for batch_idx, (data, target, _, source) in enumerate(train_loader):  # (data, target item id, target category id, idx)
        target = target if len(target) > 0 else None
        if cuda:
            data = data.cuda()
            if target is not None:
                target = target.cuda()
            source = source.cuda()

        if adv_train:
            model.eval()
            adv_data = attack.perturb(data, source, 'mean', True)
            model.train()
            optimizer.zero_grad()
            outputs, _ = model(adv_data)
        else:
            model.train()
            optimizer.zero_grad()
            outputs = model(data)

        # loss_inputs = (outputs, target, )
        loss_inputs = (outputs[0], target, )
        loss_outputs = loss_fn(*loss_inputs)
        loss_sim = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses['sim'].append(loss_sim.item())
        total_loss_sim += loss_sim.item()

        loss_mix_div = - sum(torch.sum(outputs[1] * torch.log(outputs[1]), dim=1)) / outputs[1].shape[0]
        losses['mixture_div'].append(loss_mix_div.item())
        total_loss_mix_div += loss_mix_div.item()

        loss = loss_sim + loss_mix_div
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            if metric.name != 'Retrieval Accuracy':
                metric(outputs, target, loss_outputs)
                writer.add_scalars('Metric/{}'.format(metric.name()), {'train': metric.value()}, num_iter + batch_idx + 1)
            else:
                metric(outputs, target, source)

        writer.add_scalars('Loss/total', {'train': loss.item()}, num_iter + batch_idx + 1)
        writer.add_scalars('Loss/similarity', {'train': loss_sim.item()}, num_iter + batch_idx + 1)
        writer.add_scalars('Loss/miture-divergence', {'train': loss_mix_div.item()}, num_iter + batch_idx + 1)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss-Sim: {:.6f}\tLoss-MixtureDivergence: {:.6f}'.format(
                batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader), np.mean(losses['sim']), np.mean(losses['mixture_div']))
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

            print(message)
            losses = {'sim': [], 'mixture_div': []}

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        val_loss_sim = 0
        val_loss_mix_div = 0
        for batch_idx, (data, target, cate, source) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
                source = source.cuda()

            # outputs, _ = model(*data)
            outputs = model(*data)

            loss_inputs = (outputs[0], target, )
            loss_outputs = loss_fn(*loss_inputs)
            loss_sim = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss_sim += loss_sim.item()

            loss_mix_div = - sum(torch.sum(outputs[1] * torch.log(outputs[1]), dim=1)) / outputs[1].shape[0]
            val_loss_mix_div += loss_mix_div.item()

            loss = loss_sim + loss_mix_div
            val_loss += loss.item()

            for metric in metrics:
                if metric.name == 'Retrieval Accuracy':
                    metric(outputs, target, source)
                else:
                    metric(outputs, target, loss_outputs)

    return val_loss, val_loss_sim, val_loss_mix_div, metrics


def train_domain_classifier_epoch(train_loader, model, attack, loss_fn, criterion, optimizer, cuda, writer, epoch,
                                  log_interval, metrics, unsup_da=False):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = {'sim': [], 'domain_cls': []}
    total_loss = 0
    total_loss_sim = 0
    total_loss_domain_cls = 0
    num_iter = epoch * len(train_loader)

    for batch_idx, (data, target, _, source) in enumerate(train_loader):
        # (data, target item id, target category id, idx)
        target = target if len(target) > 0 else None
        if cuda:
            data = data.cuda()
            if target is not None:
                target = target.cuda()
            source = source.cuda()

        if attack is not None:
            model.eval()
            adv_data = attack.perturb(data, source, 'mean', True)
            optimizer.zero_grad()
            outputs = model(adv_data)
        else:
            optimizer.zero_grad()
            outputs = model(data)

        if unsup_da:
            loss_inputs = (outputs[0][source == 1], target[source == 1], )
        else:
            loss_inputs = (outputs[0], target, )

        loss_outputs = loss_fn(*loss_inputs)
        loss_sim = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses['sim'].append(loss_sim.item())

        loss_domain_cls = criterion(outputs[1], source)
        losses['domain_cls'].append(loss_domain_cls.item())

        loss = loss_sim + loss_domain_cls
        total_loss += loss.item()
        total_loss_sim += loss_sim.item()
        total_loss_domain_cls += loss_domain_cls.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)
            writer.add_scalars('Metric/{}'.format(metric.name()), {'train': metric.value()}, num_iter + batch_idx + 1)

        writer.add_scalars('Loss/similarity', {'train': loss_sim.item()}, num_iter + batch_idx + 1)
        writer.add_scalars('Loss/domain_clf', {'train': loss_domain_cls.item()}, num_iter + batch_idx + 1)
        writer.add_scalars('Loss/total', {'train': loss.item()}, num_iter + batch_idx + 1)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss Sim: {:.6f} Domain: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses['sim']), np.mean(losses['domain_cls']))
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

            print(message)
            losses = {'sim': [], 'domain_cls': []}

    total_loss /= (batch_idx + 1)
    total_loss_sim /= (batch_idx + 1)
    total_loss_domain_cls /= (batch_idx + 1)
    return total_loss, total_loss_sim, total_loss_domain_cls, metrics


def test_domain_classifier_epoch(val_loader, model, loss_fn, criterion, cuda, metrics, unsup_da=False):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        val_loss_sim = 0
        val_loss_domain_cls = 0
        for batch_idx, (data, target, _, source) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
                source = source.cuda()

            outputs = model(*data)

            if unsup_da:
                loss_inputs = (outputs[0][source == 1], target[source == 1],)
            else:
                loss_inputs = (outputs[0], target,)

            loss_outputs = loss_fn(*loss_inputs)
            loss_sim = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            loss_domain_cls = criterion(outputs[1], source)
            loss = loss_sim + loss_domain_cls

            val_loss += loss.item()
            val_loss_sim += loss_sim.item()
            val_loss_domain_cls += loss_domain_cls.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, val_loss_sim, val_loss_domain_cls, metrics

