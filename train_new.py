import numpy as np
import argparse, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim

from datasets import BalancedBatchSampler, DeepFashionDataset
from networks import ResNetbasedNet
from losses import OnlineTripletLoss
from utils import read_data, RandomNegativeTripletSelector, HardestNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from trainer import fit
from inference import get_topK_acc, get_embedding_mtx

# from detectron2.config import get_cfg


# sys.path.append('/home/jayeon/Documents/code/siamese-triplet')


def main(args):
    # config_path = "/home/jayeon/Documents/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    # cfg = get_cfg()
    # cfg.merge_from_file(config_path)
    # cfg.MODEL.WEIGHTS = "/home/jayeon/Documents/detectron2/tools/output_integrated/model_final.pth"
    # cfg.MODEL.BACKBONE.FREEZE_AT = 1
    model_save_dir = args.model_path

    img_list, base_path, item_dict = read_data("DeepFashion2", bbox_gt=True)

    model = ResNetbasedNet()
    # model.load_state_dict(torch.load('/home/jayeon/Documents/siamese-triplet/model_norm/00069.pth'))
    # model = ResNetbasedNet(cfg)

    is_cud = torch.cuda.is_available()
    device = torch.device("cuda" if is_cud else "cpu")
    if is_cud:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    kwargs = {'num_workers': 8, 'pin_memory': True} if is_cud else {}

    if args.phase == 'train':
        train_dataset = DeepFashionDataset(img_list['train'], root=base_path, augment=True)
        train_batch_sampler = BalancedBatchSampler(train_dataset.labels, train_dataset.source, n_classes=100, n_samples=2)
        online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)

        test_dataset = DeepFashionDataset(img_list['validation'], root=base_path)
        test_batch_sampler = BalancedBatchSampler(test_dataset.labels, test_dataset.source, n_classes=100, n_samples=2)
        online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

        margin = 0.4
        loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin))
        grad_norm = 1.
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        # nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.5, last_epoch=-1)
        n_epochs = 200
        log_interval = 200

        fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, is_cud, log_interval,
            model_save_dir, metrics=[AverageNonzeroTripletsMetric()], start_epoch=0)

    else:
        model.eval()
        test_dataset = DeepFashionDataset(img_list['validation'], root=base_path)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
        embedding_mtx = np.zeros((len(test_dataset), 128))
        top_k = 10000
        idx_ = 0
        start_time = time.time()
        for idx, (data, target) in enumerate(test_loader):
            emb_vecs = model(data.cuda())
            embedding_mtx[idx_: idx_ + len(data)] = emb_vecs.cpu().numpy()
            idx_ += len(data)
            if idx % 20 == 0:
                print(
                    'processing {}/{}... elapsed time {}s'.format(idx + 1, len(test_loader), time.time() - start_time))

        labels = np.asarray([data_[1] for data_ in test_dataset], dtype=int)
        result_arr = np.zeros((len(test_dataset), top_k))
        for idx, vec in enumerate(embedding_mtx):
            result_arr[idx] = np.delete(np.argsort(((vec - embedding_mtx) ** 2).mean(axis=1) ** 0.5), idx)[:top_k]
            result_arr[idx] = labels[result_arr[idx]] == labels[idx]

        np.save(args.result_path, result_arr)

        for k in [1, 5, 10, 20, 100, 200, 500]:
            topk_accuracy = np.sum(np.sum(result_arr[:, :k], axis=1) > 0) / result_arr.shape[0]
            print('Top-{} Accuracy: {:.5f}'.format(k, topk_accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image retrieval system')
    parser.add_argument('-d', '--dataset', required=True, help='DeepFashion / DeepFashion2')
    # parser.add_argument('-b', '--bbox', required=True, help='gt / pred')
    parser.add_argument('--phase', required=False, default='train', help='Train or Inference')
    parser.add_argument('-mp', '--model_path', required=True)
    parser.add_argument('-r', dest='result_path', required=False, help='Directory to save the results')
    args = parser.parse_args()

    main(args)
