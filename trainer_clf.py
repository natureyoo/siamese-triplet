import torch
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter


def fit(train_loader, val_loader, model, criterion1, criterion2, optimizer, scheduler, n_epochs, cuda, log_interval,
        model_save_dir, start_epoch=0):
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

    writer = SummaryWriter('runs/clf_128')

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_loss = train_epoch(train_loader, model, criterion1, criterion2, optimizer, cuda, writer, epoch, log_interval)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f} Elapsed time: {}s'\
            .format(epoch + 1, n_epochs, train_loss, int(time.time() - start_time))

        scheduler.step()
        val_loss = test_epoch(val_loader, model, criterion2, cuda)
        val_loss /= len(val_loader)
        writer.add_scalars('Loss/category_clf', {'validation': val_loss}, (epoch + 1) * len(train_loader) - 1)

        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
        torch.save(model.module.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(str(epoch).zfill(5))))

        message += '\nEpoch: {}/{}. Validation set: Average Category loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)
        print(message)


def train_epoch(train_loader, model, criterion1, criterion2, optimizer, cuda, writer, epoch, log_interval):
    model.train()
    losses1 = []
    losses2 = []
    total_loss = 0
    num_iter = epoch * len(train_loader)

    for batch_idx, (data, label, cate, _) in enumerate(train_loader):    # (data, target item id, target category id, idx)
        if cuda:
            data = data.cuda()
            label = label.cuda()
            cate = cate.cuda()

        optimizer.zero_grad()
        outputs = model(data)

        loss1 = criterion1(outputs[0], label)
        loss2 = criterion2(outputs[1], cate)
        loss = loss1 + loss2

        losses1.append(loss1.item())
        losses2.append(loss2.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        writer.add_scalars('Loss/instance_clf', {'train': loss1.item()}, num_iter + batch_idx + 1)
        writer.add_scalars('Loss/category_clf', {'train': loss2.item()}, num_iter + batch_idx + 1)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss-Instance: {:.6f}\tLoss-Category: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses1), np.mean(losses2))

            print(message)
            losses1 = []
            losses2 = []

    total_loss /= (batch_idx + 1)
    return total_loss


def test_epoch(val_loader, model, criterion2, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, _, cate, _) in enumerate(val_loader):
            if cuda:
                data = data.cuda()
                cate = cate.cuda()

            outputs = model(data)
            loss = criterion2(outputs[1], cate)

            val_loss += loss.item()

    return val_loss
