import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import BalancedBatchSampler, DeepFashionDataset
import torch.optim as optim
from torch.optim import lr_scheduler

# from torch.utils.tensorboard import SummaryWriter

from networks import ResNetbasedNet
from losses import OnlineTripletLoss
from utils import read_data, RandomNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from trainer import fit
from inference import get_topK_acc
import numpy as np
import os
import argparse
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.layers import ShapeSpec

# sys.path.append('/home/jayeon/Documents/code/siamese-triplet')


def main(args):
    if os.path.exists('models') is False:
        os.makedirs('models')

    # img_list, base_path, item_dict = read_data(args.dataset, args.bbox)
    img_list, base_path, item_dict = read_data("DeepFashion2", bbox_gt=False)
    model_save_path = args.model_path   # 'models/siames_triplet_df2.pth'

    # writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    model = ResNetbasedNet()
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}

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
    parser.add_argument('-b', '--bbox', required=True, help='gt / pred')
    parser.add_argument('--phase', required=False, default='train', help='Train or Inference')
    parser.add_argument('-mp', '--model_path', required=True)
    args = parser.parse_args()

    main(args)
