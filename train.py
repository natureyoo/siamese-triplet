import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import BalancedBatchSampler, DeepFashionDataset
from torch.optim import lr_scheduler
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import torchvision

from networks import EmbeddingNet, ResNet50basedNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from utils import read_data
from metrics import AverageNonzeroTripletsMetric
from trainer import fit
from inference import get_topK_acc
import numpy as np
import os
import sys
import argparse
import detectron2

# sys.path.append('/home/jayeon/Documents/code/siamese-triplet')


def main(args):
    if os.path.exists('models') is False:
        os.makedirs('models')

    dataset_type = args.dataset   # '/second/DeepFashion/'
    img_list, base_path, item_dict = read_data(dataset_type)

    model_save_path = args.model_path   # 'models/siames_triplet_df2.pth'

    # writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    if os.path.exists(model_save_path):
        backbone = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
        model = ResNet50basedNet(backbone)
        model.load_state_dict(torch.load(model_save_path))
    else:
        # backbone = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        backbone = detectron2.
        model = ResNet50basedNet(backbone)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    if not args.phase:
        train_dataset = DeepFashionDataset(img_list['train'], root=base_path)
        train_batch_sampler = BalancedBatchSampler(train_dataset.labels, train_dataset.source, n_classes=32, n_samples=4)
        online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)

        test_dataset = DeepFashionDataset(img_list['validation'], root=base_path)
        test_batch_sampler = BalancedBatchSampler(test_dataset.labels, test_dataset.source, n_classes=32, n_samples=4)
        online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

        margin = 1.
        loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
        lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
        n_epochs = 20
        log_interval = 200

        fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
            model_save_path, metrics=[AverageNonzeroTripletsMetric()])

    else:
        model.eval()
        gallery_dataset = DeepFashionDataset(img_list['validation'], root=base_path)

        acc, query_cnt = get_topK_acc(gallery_dataset, item_dict['validation'], model, cuda, open('retrieval_result.txt', 'a'), 100)
        np.savetxt('TopK_accuracy.txt', np.concatenate((acc, np.asarray([query_cnt]))), fmt='%1.5f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image retrieval system')
    parser.add_argument('-d', '--dataset', required=True, help='DeepFashion / DeepFashion2')
    parser.add_argument('--phase', required=False, default='train', help='Train or Inference')
    parser.add_argument('-mp', '--model_path', required=True)
    args = parser.parse_args()

    main(args)
