import numpy as np
import argparse, time, os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim

from datasets import BalancedBatchSampler, DeepFashionDataset
from networks import ResNetbasedNet
from losses import OnlineTripletLoss, AllTripletLoss
from utils import read_data, pdist, RandomNegativeTripletSelector, HardestNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from trainer import fit
# from trainer_trip_clf import fit
# from inference import get_topK_acc, get_embedding_mtx


def main(args):
    model_path = args.model_path
    save_dir = args.save_dir
    vec_dim = 128

    data_type = ['validation'] if args.phase == 'test' else ['train', 'validation']
    img_list, base_path, item_dict = read_data("DeepFashion2", bbox_gt=True, type_list=data_type)

    model = ResNetbasedNet(vec_dim=vec_dim, max_pool=True, load_path=model_path)

    is_cud = torch.cuda.is_available()
    device = torch.device("cuda" if is_cud else "cpu")
    if is_cud:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    kwargs = {'num_workers': 8, 'pin_memory': True} if is_cud else {}

    if args.phase == 'train':
        train_dataset = DeepFashionDataset(img_list['train'], root=base_path, augment=True)
        train_batch_sampler = BalancedBatchSampler(train_dataset.labels, train_dataset.source, n_classes=64, n_samples=4)
        online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)

        test_dataset = DeepFashionDataset(img_list['validation'], root=base_path)
        test_batch_sampler = BalancedBatchSampler(test_dataset.labels, test_dataset.source, n_classes=64, n_samples=4)
        online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

        margin = 0.2
        # loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin))
        loss_fn = AllTripletLoss(margin)
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, threshold=0.001, cooldown=2, min_lr=1e-5 / (10 * 2),)
        n_epochs = 100
        log_interval = 200
        n_train_data_len = len(train_dataset.labels)

        fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, n_train_data_len,
            is_cud, log_interval, save_dir, metrics=[AverageNonzeroTripletsMetric()], start_epoch=0)

    else:
        with torch.no_grad():
            model.eval()
            test_dataset = DeepFashionDataset(img_list['validation'], root=base_path)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
            embedding_mtx = torch.zeros((len(test_dataset), vec_dim))
            labels = np.zeros(len(test_dataset))
            top_k = 500
            idx_ = 0
            start_time = time.time()
            cf_mtx = np.zeros(4, dtype=float) # predict_user_real_user / predict_user_real_shop / predict_shop_real_user / predict_shop_real_shop

            for idx, (data, target, _, source) in enumerate(test_loader):
                emb_vecs = model(data.cuda())
                embedding_mtx[idx_: idx_ + len(data)] = emb_vecs[0]
                predict = torch.argmax(emb_vecs[1], dim=1).cpu().numpy()
                real = source.cpu().numpy()
                cf_mtx[0] += np.sum((predict == 0) & (real == 0))
                cf_mtx[1] += np.sum((predict == 0) & (real == 1))
                cf_mtx[2] += np.sum((predict == 1) & (real == 0))
                cf_mtx[3] += np.sum((predict == 1) & (real == 1))
                labels[idx_:idx_ + len(data)] = np.asarray(target)
                idx_ += len(data)
                if idx % 20 == 0:
                    print(
                        'processing {}/{}... elapsed time {}s'.format(idx + 1, len(test_loader), time.time() - start_time))

        print('Total: {}, Domain Classification Acc: {:.5f}'.format(np.sum(cf_mtx), (cf_mtx[0] + cf_mtx[3]) / np.sum(cf_mtx)))
        print('Recall User Photo: {:.5f}'.format(cf_mtx[0] / (cf_mtx[0] + cf_mtx[2])))
        print('Recall Shop Photo: {:.5f}'.format(cf_mtx[3] / (cf_mtx[1] + cf_mtx[3])))

        np.save(os.path.join(save_dir, 'emb_mtx.npy'), embedding_mtx)
        with open(os.path.join(save_dir, 'file_info.txt'), 'w') as f:
            for i in range(len(test_dataset)):
                f.write('{},{},{},{}\n'.format(img_list['validation'][i][0], test_dataset[i][1], test_dataset[i][2], test_dataset[i][3]))
        print('save files!')

        distance_mtx = pdist(embedding_mtx)
        sorted_idx = torch.argsort(distance_mtx, dim=1).cpu().numpy()
        result_arr = np.zeros((sorted_idx.shape[0], top_k))
        for idx in range(sorted_idx.shape[0]):
            result_arr[idx] = sorted_idx[idx][sorted_idx[idx] != idx][:top_k]
            result_arr[idx] = labels[result_arr[idx].astype(np.int)] == labels[idx]
            if idx % 1000 == 0:
                print(idx)

        for k in [1, 5, 10, 20, 100, 200, 500]:
            topk_accuracy = np.sum(np.sum(result_arr[:, :k], axis=1) > 0) / result_arr.shape[0]
            print('Top-{} Accuracy: {:.5f}'.format(k, topk_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image retrieval system')
    parser.add_argument('-d', '--dataset', required=True, help='DeepFashion / DeepFashion2')
    # parser.add_argument('-b', '--bbox', required=True, help='gt / pred')
    parser.add_argument('--phase', required=False, default='train', help='Train or Inference')
    parser.add_argument('-mp', dest='model_path', required=False, default=None, help='pretrained model path')
    parser.add_argument('-s', dest='save_dir', required=True, help='Directory to save the models and the results')
    args = parser.parse_args()

    main(args)
