import torch
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, model_save_dir,
        metrics=[], start_epoch=0, domain_adap=False):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    best_val_loss = None
    start_time = time.time()

    # for epoch in range(0, start_epoch):
    #     scheduler.step()
    writer = SummaryWriter('runs/clf128_based')

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, writer, epoch, log_interval, metrics, domain_adap)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f} Elapsed time: {}s'\
            .format(epoch + 1, n_epochs, train_loss, int(time.time() - start_time))
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

        scheduler.step()
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics, domain_adap)
        val_loss /= len(val_loader)
        writer.add_scalars('Loss/train+val', {'validation': val_loss}, (epoch + 1) * len(train_loader) - 1)

        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
        torch.save(model.module.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(str(epoch).zfill(5))))

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            writer.add_scalars('Metric/{}'.format(metric.name()), {'validation': metric.value()}, (epoch + 1) * len(train_loader) - 1)

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, writer, epoch, log_interval, metrics, domain_adap):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    num_iter = epoch * len(train_loader)

    for batch_idx, (data, target, source) in enumerate(train_loader):    # (data, target item id, target category id, idx)
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
            if domain_adap:
                source = source.cuda()

        optimizer.zero_grad()
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
