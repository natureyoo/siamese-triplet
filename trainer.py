import torch
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
from attack import FastGradientSignUntargeted


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, model_save_dir,
        metrics=[], start_epoch=0, criterion=None, domain_adap=False, adv_train=False, adv_epsilon=None, adv_alph=None, adv_iter=None):

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

        # train_loss, metrics = train_epoch(train_loader, model, attack, loss_fn, optimizer, cuda, writer, epoch,
        #                                   log_interval, metrics, adv_train)
        # message = 'Epoch: {}/{}. Train set: Average loss: {:.4f} Elapsed time: {}s'\
        #     .format(epoch + 1, n_epochs, train_loss, int(time.time() - start_time))

        train_loss, train_loss_sim, train_loss_domain_cls, metrics = train_domain_classifier_epoch(train_loader, model,
                                    attack, loss_fn, criterion, optimizer, cuda, writer, epoch, log_interval, metrics)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f} loss-sim: {:.4f} loss-domain-cls: {:.4f} Elapsed time: {}s'\
            .format(epoch + 1, n_epochs, train_loss, train_loss_sim, train_loss_domain_cls, int(time.time() - start_time))

        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

        # val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics, domain_adap)
        val_loss, val_loss_sim, val_loss_domain_cls, metrics = test_domain_classifier_epoch(val_loader, model,
                                                                        loss_fn, criterion, cuda, metrics)
        val_loss /= len(val_loader)
        val_loss_sim /= len(val_loader)
        val_loss_domain_cls /= len(val_loader)
        writer.add_scalars('Loss/total', {'validation': val_loss}, (epoch + 1) * len(train_loader) - 1)
        # message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}' \
        #     .format(epoch + 1, n_epochs, val_loss)
        writer.add_scalars('Loss/similarity', {'validation': val_loss_sim}, (epoch + 1) * len(train_loader) - 1)
        writer.add_scalars('Loss/domain_clf', {'validation': val_loss_domain_cls}, (epoch + 1) * len(train_loader) - 1)
        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f} loss-sim: {:.4f} loss-domain-cls: {:.4f}'\
            .format(epoch + 1, n_epochs, val_loss, val_loss_sim, val_loss_domain_cls)

        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
        torch.save(model.module.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(str(epoch).zfill(5))))

        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            writer.add_scalars('Metric/{}'.format(metric.name()), {'validation': metric.value()}, (epoch + 1) * len(train_loader) - 1)
        # scheduler.step(val_loss)
        scheduler.step(metrics[0].value())
        print(message)


def train_epoch(train_loader, model, attack, loss_fn, optimizer, cuda, writer, epoch, log_interval, metrics, adv_train):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    num_iter = epoch * len(train_loader)

    for batch_idx, (data, target, _, source) in enumerate(train_loader):    # (data, target item id, target category id, idx)
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

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target
        # loss_inputs += (source, )
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)
            writer.add_scalars('Metric/{}'.format(metric.name()), {'train': metric.value()}, num_iter + batch_idx + 1)

        writer.add_scalars('Loss/train+val', {'train': loss.item()}, num_iter + batch_idx + 1)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics, domain_adap):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target, cate, source) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
                if domain_adap:
                    source = source.cuda()

            # outputs, _ = model(*data)
            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
            if domain_adap:
                loss_inputs += (source, )

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


def train_domain_classifier_epoch(train_loader, model, attack, loss_fn, criterion, optimizer, cuda, writer, epoch,
                                  log_interval, metrics):
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

        loss_inputs = (outputs[0], )
        if target is not None:
            target = (target,)
            loss_inputs += target

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


def test_domain_classifier_epoch(val_loader, model, loss_fn, criterion, cuda, metrics):
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

            loss_inputs = (outputs[0], )
            if target is not None:
                target = (target,)
                loss_inputs += target

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

