import torch
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, n_train_data_len,
        cuda, log_interval, model_save_dir, metrics=[], start_epoch=0):

    best_val_loss = None
    start_time = time.time()

    # for epoch in range(0, start_epoch):
    #     scheduler.step()
    writer = SummaryWriter('runs/{}'.format(model_save_dir))

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, writer, epoch,
                                          log_interval, n_train_data_len, metrics)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f} Elapsed time: {}s'\
            .format(epoch + 1, n_epochs, train_loss, int(time.time() - start_time))

        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

        summ_step = (epoch + 1) * len(train_loader) - 1

        # Test stage
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        writer.add_scalars('Loss/total', {'validation': val_loss}, summ_step)
        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)

        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.module.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(str(epoch).zfill(5))))

        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            writer.add_scalars('Metric/{}'.format(metric.name()), {'validation': metric.value()}, summ_step)

        scheduler.step(val_loss)
        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, writer, epoch, log_interval, n_train_data_len, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    num_iter = epoch * len(train_loader)
    cur_iter = 0
    for batch_idx, (data, target, _, _) in enumerate(train_loader):  # (data, target item id, target category id, idx)
        cur_iter += len(data)
        target = target if len(target) > 0 else None
        if cuda:
            data = data.cuda()
            if target is not None:
                target = target.cuda()

            optimizer.zero_grad()
            outputs = model(data)

        loss_inputs = (outputs, target, )
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        writer.add_scalars('Loss/total', {'train': loss.item()}, num_iter + batch_idx + 1)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                cur_iter, n_train_data_len, 100. * cur_iter / n_train_data_len, np.mean(losses))
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)

    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target, _, _) in enumerate(val_loader):
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            loss_inputs = (outputs, target, )
            loss_outputs = loss_fn(*loss_inputs)
            loss= loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
