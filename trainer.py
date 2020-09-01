import torch
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
from attack import FastGradientSignUntargeted


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, model_save_dir,
        metrics=[], start_epoch=0, criterion=None, use_dt=False, adv_train=False, adv_epsilon=None,
        adv_alph=None, adv_iter=None, clf_task=None):

    best_val_loss = None
    start_time = time.time()

    for epoch in range(0, start_epoch):
        scheduler.step()

    writer = SummaryWriter('runs/{}'.format(model_save_dir))

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        if adv_train:
            attack = FastGradientSignUntargeted(model, adv_epsilon, adv_alph, min_val=-2.5, max_val=3.0,
                                                max_iters=adv_iter, _type='linf')  # l2
        else:
            attack = None

        if clf_task is None:
            train_loss, metrics = train_epoch(train_loader, model, attack, loss_fn, optimizer, cuda, writer,
                                              epoch, log_interval, metrics, adv_train)
            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f} Elapsed time: {}s'\
                .format(epoch + 1, n_epochs, train_loss, int(time.time() - start_time))

        else:
            train_loss, train_loss_sim, train_loss_cls, metrics = train_with_classifier_epoch(train_loader, model,
                                attack, loss_fn, criterion, optimizer, cuda, writer, epoch, log_interval, metrics,
                                use_dt=use_dt, clf_task=clf_task)
            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f} loss-sim: {:.4f} loss-cls: {:.4f} Elapsed time: {}s'\
                .format(epoch + 1, n_epochs, train_loss, train_loss_sim, train_loss_cls, int(time.time() - start_time))

        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

        summ_step = (epoch + 1) * len(train_loader) - 1

        # Test stage
        if clf_task is None:
            val_loss, val_loss_sim, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        else:
            val_loss, val_loss_sim, val_loss_cls, metrics = test_domain_classifier_epoch(val_loader, model,
                                            loss_fn, criterion, cuda, metrics, use_dt=use_dt, clf_task=clf_task)
        val_loss /= len(val_loader)
        val_loss_sim /= len(val_loader)
        writer.add_scalars('Loss/total', {'validation': val_loss}, summ_step)
        writer.add_scalars('Loss/similarity', {'validation': val_loss_sim}, summ_step)
        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f} loss-sim: {:.4f}'\
                .format(epoch + 1, n_epochs, val_loss, val_loss_sim)

        if criterion is not None:
            val_loss_cls /= len(val_loader)
            writer.add_scalars('Loss/clf', {'validation': val_loss_cls}, summ_step)
            message += ' loss-cls: {:.4f}'.format(val_loss_cls)

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
    losses = {'sim': [], 'mixture_div': [], 'cate_cls': []}
    total_loss = 0
    total_loss_sim = 0
    total_loss_mix_div = 0
    total_loss_cate_cls = 0
    num_iter = epoch * len(train_loader)

    for batch_idx, (data, target, cate, source) in enumerate(train_loader):  # (data, target item id, target category id, idx)
        target = target if len(target) > 0 else None
        if cuda:
            data = data.cuda()
            if target is not None:
                target = target.cuda()
            source = source.cuda()
            cate = cate.cuda()

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

        loss_inputs = (outputs, target, )
        # loss_inputs = (outputs[0], target, )
        loss_outputs = loss_fn(*loss_inputs)
        loss_sim = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses['sim'].append(loss_sim.item())
        total_loss_sim += loss_sim.item()

        # loss_mix_div = - sum(torch.sum(outputs[2] * torch.log(outputs[2]), dim=1)) / outputs[2].shape[0]
        # losses['mixture_div'].append(loss_mix_div.item())
        # total_loss_mix_div += loss_mix_div.item()
        #
        # loss_cate_cls = criterion(outputs[1], cate)
        # losses['cate_cls'].append(loss_cate_cls.item())
        # total_loss_cate_cls += loss_cate_cls.item()

        loss = loss_sim
        # loss = loss_sim + loss_mix_div + loss_cate_cls
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)
            writer.add_scalars('Metric/{}'.format(metric.name()), {'train': metric.value()}, num_iter + batch_idx + 1)

        writer.add_scalars('Loss/total', {'train': loss.item()}, num_iter + batch_idx + 1)
        writer.add_scalars('Loss/similarity', {'train': loss_sim.item()}, num_iter + batch_idx + 1)
        # writer.add_scalars('Loss/miture-divergence', {'train': loss_mix_div.item()}, num_iter + batch_idx + 1)
        # writer.add_scalars('Loss/category', {'train': loss_cate_cls.item()}, num_iter + batch_idx + 1)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss-Sim: {:.6f}'.format(batch_idx * len(data),
                len(train_loader) * len(data), 100. * batch_idx / len(train_loader), np.mean(losses['sim']))
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

            print(message)
            losses = {'sim': [], 'mixture_div': [], 'cate_cls': []}

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
        val_loss_cate = 0
        for batch_idx, (data, target, cate, source) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
                source = source.cuda()
                cate = cate.cuda()

            # outputs, _ = model(*data)
            outputs = model(*data)

            loss_inputs = (outputs, target, )
            # loss_inputs = (outputs[0], target, )
            loss_outputs = loss_fn(*loss_inputs)
            loss_sim = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss_sim += loss_sim.item()

            # loss_mix_div = - sum(torch.sum(outputs[2] * torch.log(outputs[2]), dim=1)) / outputs[2].shape[0]
            # val_loss_mix_div += loss_mix_div.item()
            #
            # loss_cate = criterion(outputs[1], cate)
            # val_loss_cate + loss_cate.item()

            loss = loss_sim
            # loss = loss_sim + loss_mix_div + loss_cate
            val_loss += loss.item()

            for metric in metrics:
                if metric.name == 'Retrieval Accuracy':
                    metric(outputs, target, source)
                else:
                    metric(outputs, target, loss_outputs)

    # return val_loss, val_loss_sim, val_loss_mix_div, val_loss_cate, metrics
    return val_loss, val_loss_sim, metrics


def train_with_classifier_epoch(train_loader, model, attack, loss_fn, criterion, optimizer, cuda, writer, epoch,
                                  log_interval, metrics, use_dt=False, clf_task='category'):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = {'sim': [], 'cls': []}
    total_loss = 0
    total_loss_sim = 0
    total_loss_cls = 0
    num_iter = epoch * len(train_loader)

    for batch_idx, (data, target, cate, source) in enumerate(train_loader):
        # (data, target item id, target category id, idx)
        target = target if len(target) > 0 else None
        if cuda:
            data = data.cuda()
            if target is not None:
                target = target.cuda()
            cate = cate.cuda()
            source = source.cuda()
        clf_label = cate if clf_task == 'category' else source

        if attack is not None:
            model.eval()
            adv_data = attack.perturb(data, source, 'mean', True)
            optimizer.zero_grad()
            outputs = model(adv_data)
        else:
            optimizer.zero_grad()
            outputs = model(data)

        if use_dt:
            loss_inputs = (outputs[0][source == 1], target[source == 1], )
        else:
            loss_inputs = (outputs[0], target, )

        loss_outputs = loss_fn(*loss_inputs)
        loss_sim = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses['sim'].append(loss_sim.item())

        loss_cls = criterion(outputs[1], clf_label)
        losses['cls'].append(loss_cls.item())

        loss = loss_sim + loss_cls
        total_loss += loss.item()
        total_loss_sim += loss_sim.item()
        total_loss_cls += loss_cls.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)
            writer.add_scalars('Metric/{}'.format(metric.name()), {'train': metric.value()}, num_iter + batch_idx + 1)

        writer.add_scalars('Loss/similarity', {'train': loss_sim.item()}, num_iter + batch_idx + 1)
        writer.add_scalars('Loss/domain_clf', {'train': total_loss_cls.item()}, num_iter + batch_idx + 1)
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
    total_loss_cls /= (batch_idx + 1)
    return total_loss, total_loss_sim, total_loss_cls, metrics


def test_domain_classifier_epoch(val_loader, model, loss_fn, criterion, cuda, metrics, use_dt=False, clf_task='category'):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        val_loss_sim = 0
        val_loss_cls = 0
        for batch_idx, (data, target, cate, source) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
                cate = cate.cuda()
                source = source.cuda()
            clf_label = cate if clf_task == 'category' else source

            outputs = model(*data)

            if use_dt:
                loss_inputs = (outputs[0][source == 1], target[source == 1],)
            else:
                loss_inputs = (outputs[0], target,)

            loss_outputs = loss_fn(*loss_inputs)
            loss_sim = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            loss_cls = criterion(outputs[1], clf_label)
            loss = loss_sim + loss_cls

            val_loss += loss.item()
            val_loss_sim += loss_sim.item()
            val_loss_cls += loss_cls.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, val_loss_sim, val_loss_cls, metrics

