import numpy as np
import argparse, time, os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim

from datasets import DABatchSampler, DeepFashionDataset
from networks import ResNetbasedNet, MixtureNet
from losses import OnlineTripletLoss, AllTripletLoss
from utils import read_data, pdist, RandomNegativeTripletSelector, HardestNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric, RetrivalAccMetric
from trainer import fit
# from trainer_trip_clf import fit
# from inference import get_topK_acc, get_embedding_mtx


def eval_retrieval_acc(embedding_mtx, sources, labels):
    user_idx = np.where(sources == 0)[0]
    shop_idx = np.where(sources == 1)[0]
    user_emb_mtx = embedding_mtx[user_idx]
    shop_emb_mtx = embedding_mtx[shop_idx]

    acc = {'inshop': [], 'user2shop': []}

    # in-shop
    dist = pdist(shop_emb_mtx, shop_emb_mtx)
    sort_idx = np.argsort(dist, axis=1)[:, 1:101]

    result_arr = np.zeros((len(user_idx), 100))
    for i, idx in enumerate(user_idx):
        result_arr[i] = labels[shop_idx[sort_idx[i].astype(np.int)]] == labels[idx]

    print('Inshop Retrieval Acc')
    for k in [1, 5, 10, 20, 100]:
        topk_accuracy = np.sum(np.sum(result_arr[:, :k], axis=1) > 0) / result_arr.shape[0]
        print('Top-{} Accuracy: {:.5f}'.format(k, topk_accuracy))
        acc['inshop'].append(topk_accuracy)

    # user-shop
    dist = pdist(user_emb_mtx, shop_emb_mtx)
    sort_idx = np.argsort(dist, axis=1)[:, :100]

    result_arr = np.zeros((len(user_idx), 100))
    for i, idx in enumerate(user_idx):
        result_arr[i] = labels[shop_idx[sort_idx[i].astype(np.int)]] == labels[idx]

    print('User2shop Retrieval Acc')
    for k in [1, 5, 10, 20, 100]:
        topk_accuracy = np.sum(np.sum(result_arr[:, :k], axis=1) > 0) / result_arr.shape[0]
        print('Top-{} Accuracy: {:.5f}'.format(k, topk_accuracy))
        acc['user2shop'].append(topk_accuracy)

    return acc


def main(args):
    model_path = args.model_path
    save_dir = args.save_dir
    vec_dim = 128

    data_type = ['validation'] if args.phase == 'test' else ['train', 'validation']
    img_list, base_path, item_dict = read_data("DeepFashion2", bbox_gt=True, type_list=data_type)

    class_num = {'category': 13, 'domain': 2}
    clf_num = None if args.multi_task is None else class_num[args.multi_task]
    model = ResNetbasedNet(vec_dim=vec_dim, max_pool=True, load_path=model_path, clf_num=clf_num)

    is_cud = torch.cuda.is_available()
    device = torch.device("cuda" if is_cud else "cpu")
    if is_cud:
        print(torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0,1])
        model.to(device)
    kwargs = {'num_workers': 4, 'pin_memory': True} if is_cud else {}

    if args.phase == 'train':
        train_dataset = DeepFashionDataset(img_list['train'], root=base_path, augment=args.augment)
        train_batch_sampler = DABatchSampler(train_dataset.labels, train_dataset.source, n_classes=64, n_samples=4, use_dt=args.use_dt)
        online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)

        test_dataset = DeepFashionDataset(img_list['validation'], root=base_path)
        test_batch_sampler = DABatchSampler(test_dataset.labels, test_dataset.source, n_classes=64, n_samples=4, use_dt=args.use_dt)
        online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

        margin = 0.2
        # loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin), domain_adap)
        loss_fn = AllTripletLoss(margin)
        criterion = nn.CrossEntropyLoss() if clf_num is not None else None
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=0.001, cooldown=2, min_lr=1e-4 / (10 * 2),)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2, threshold=0.001, cooldown=2,
                                                   min_lr=1e-7, )
        n_epochs = 100
        log_interval = 200

        # fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, is_cud,
        #     log_interval, save_dir, metrics=[AverageNonzeroTripletsMetric()], start_epoch=0,
        #     domain_adap=domain_adap, adv_train=adv_train, eval_train_dataset=train_dataset, eval_test_dataset=test_dataset)
        fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, is_cud, log_interval,
            save_dir, metrics=[AverageNonzeroTripletsMetric()], start_epoch=1, criterion=criterion, use_dt=args.use_dt, clf_task=args.multi_task)

    else:
        model.eval()
        with torch.no_grad():
            test_dataset = DeepFashionDataset(img_list['validation'], root=base_path)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)
            embedding_mtx = torch.zeros((len(test_dataset), vec_dim))
            sources = np.zeros(len(test_dataset))
            labels = np.zeros(len(test_dataset))
            idx_ = 0
            start_time = time.time()

            # predict_user_real_user / predict_user_real_shop / predict_shop_real_user / predict_shop_real_shop
            clf_correct = 0
            for idx, (data, target, cate, source) in enumerate(test_loader):
                if args.multi_task is None:
                    emb_vecs = model(data.cuda())
                else:
                    emb_vecs, clf_pred = model(data.cuda())
                    clf_label = cate.cuda() if args.multi_task == 'category' else source.cuda()
                    clf_correct += (clf_label == clf_pred).sum().item()

                embedding_mtx[idx_: idx_ + len(data)] = emb_vecs
                sources[idx_: idx_ + len(data)] = source
                labels[idx_: idx_ + len(data)] = target

                idx_ += len(data)

            print('Iteration Finished! Elapsed Time {}s'.format(time.time() - start_time))

            print('Evaluation on Validation Set')
            accuracy = eval_retrieval_acc(embedding_mtx, sources, labels)

        with open(os.path.join(save_dir, 'retieval_acc.txt'), 'a') as f:
            f.write('{},val,inshop,{}\n'.format(model_path.split('/')[-1], ','.join([str(round(a, 4)) for a in accuracy['inshop']])))
            f.write('{},val,u2shop,{}\n'.format(model_path.split('/')[-1], ','.join([str(round(a, 4)) for a in accuracy['user2shop']])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image retrieval system')
    parser.add_argument('-d', '--dataset', required=True, help='DeepFashion / DeepFashion2')
    parser.add_argument('-ag', '--augment', required=False, default=False, help='Data Augmentation')
    parser.add_argument('--phase', required=False, default='train', help='Train or Inference')
    parser.add_argument('-mp', dest='model_path', required=False, default=None, help='pretrained model path')
    parser.add_argument('-s', dest='save_dir', required=True, help='Directory to save the models and the results')
    parser.add_argument('-mt', '--multi_task', required=False, default=None, help='category/domain')
    parser.add_argument('-dt', '--use_dt', required=False, default=False, help='Use Target Domain Data')
    # parser.add_argument('--domain', dest='domain_adap', required=False, default=False, help='Train or Inference')
    # parser.add_argument('--adv_train', required=False, default=False, help='Make Adversarial Sample')
    args = parser.parse_args()

    main(args)
