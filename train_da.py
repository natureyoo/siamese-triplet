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


def eval_retrieval_acc(data_loader, metric, cuda, model, message, type='Train'):
    model.eval()
    for batch_idx, (data, target, _, source) in enumerate(data_loader):
        if cuda:
            data = data.cuda()
            target = target.cuda()
            source = source.cuda()
        outputs = model(data)
        # metric(outputs[0], target, source)
        metric(outputs, target, source)

    inshop_acc, u2shop_acc = metric.value()
    message += '\n{} {}-{}: r@1:{:.4f} r@5:{:.4f} r@10:{:.4f} r@20:{:.4f}'.format(type, metric.name(), 'inshop',
                    inshop_acc[0], inshop_acc[1], inshop_acc[2], inshop_acc[3])
    message += '\n{} {}-{}: r@1:{:.4f} r@5:{:.4f} r@10:{:.4f} r@20:{:.4f}'.format(type, metric.name(), 'u2shop',
                    u2shop_acc[0], u2shop_acc[1], u2shop_acc[2], u2shop_acc[3])

    return inshop_acc, u2shop_acc, message


def main(args):
    model_path = args.model_path
    save_dir = args.save_dir
    vec_dim = 128

    # data_type = ['validation'] if args.phase == 'test' else ['train', 'validation']
    img_list, base_path, item_dict = read_data("DeepFashion2", bbox_gt=True)

    # model = ResNetbasedNet(vec_dim=vec_dim, max_pool=True, load_path=model_path, clf2_num=2, adv_eta=1e-4)
    model = ResNetbasedNet(vec_dim=vec_dim, max_pool=True, load_path=model_path)
    # model = MixtureNet(vec_dim=vec_dim, max_pool=True, load_path=model_path, n_comp=5, clf2_num=13) # category classification
    # model = MixtureNet(vec_dim=vec_dim, max_pool=True, load_path=model_path, n_comp=3)

    domain_cls = args.domain_cls
    is_cud = torch.cuda.is_available()
    device = torch.device("cuda" if is_cud else "cpu")
    if is_cud:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    kwargs = {'num_workers': 8, 'pin_memory': True} if is_cud else {}

    if args.phase == 'train':
        train_dataset = DeepFashionDataset(img_list['train'], root=base_path, augment=True)
        train_batch_sampler = DABatchSampler(train_dataset.labels, train_dataset.source, n_classes=64, n_samples=4, domain_cls=domain_cls)
        online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)

        test_dataset = DeepFashionDataset(img_list['validation'], root=base_path)
        test_batch_sampler = DABatchSampler(test_dataset.labels, test_dataset.source, n_classes=64, n_samples=4, domain_cls=domain_cls)
        online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

        margin = 0.2
        # loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin), domain_adap)
        loss_fn = AllTripletLoss(margin)
        criterion = nn.CrossEntropyLoss() if domain_cls else None
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=0.001, cooldown=2, min_lr=1e-4 / (10 * 2),)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, threshold=1, cooldown=2, min_lr=1e-5 / (10 * 2),)
        n_epochs = 100
        log_interval = 200

        # fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, is_cud,
        #     log_interval, save_dir, metrics=[AverageNonzeroTripletsMetric()], start_epoch=0,
        #     domain_adap=domain_adap, adv_train=adv_train, eval_train_dataset=train_dataset, eval_test_dataset=test_dataset)
        fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, is_cud, log_interval,
            save_dir, metrics=[AverageNonzeroTripletsMetric()], start_epoch=0, criterion=criterion, domain_cls=domain_cls,
            unsup_da=True)

    else:
        model.eval()
        with torch.no_grad():
            test_dataset = DeepFashionDataset(img_list['validation'], root=base_path)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)
            embedding_mtx = torch.zeros((len(test_dataset), vec_dim))
            labels = np.zeros(len(test_dataset))
            idx_ = 0
            start_time = time.time()
            # predict_user_real_user / predict_user_real_shop / predict_shop_real_user / predict_shop_real_shop
            # cf_mtx = np.zeros(4, dtype=float)
            p_max = np.zeros(5)
            cate_correct = 0
            for idx, (data, target, cate, source) in enumerate(test_loader):
                if not domain_cls:
                    # emb_vecs, pred_cate, pi = model(data.cuda())
                    # emb_vecs, pi = model(data.cuda())
                    emb_vecs = model(data.cuda())
                else:
                    emb_vecs, pred_domain, pi = model(data.cuda())
                embedding_mtx[idx_: idx_ + len(data)] = emb_vecs
                # cate_correct += sum(cate == torch.argmax(pred_cate.cpu(), dim=1)).item()
                # p_cnt = np.unique(np.argmax(pi.cpu().numpy(), axis=1), return_counts=True)
                # p_max[p_cnt[0]] += p_cnt[1]
                # predict = torch.argmax(emb_vecs[1], dim=1).cpu().numpy()
                # real = source.cpu().numpy()
                # cf_mtx[0] += np.sum((predict == 0) & (real == 0))
                # cf_mtx[1] += np.sum((predict == 0) & (real == 1))
                # cf_mtx[2] += np.sum((predict == 1) & (real == 0))
                # cf_mtx[3] += np.sum((predict == 1) & (real == 1))
                labels[idx_:idx_ + len(data)] = np.asarray(target)
                idx_ += len(data)

            print('Iteration Finished! Elapsed Time {}s'.format(idx + 1, len(test_loader), time.time() - start_time))
            # p_max /= np.sum(p_max)
            # print('Total: {}, Category Acc: {:.5f}'.format(len(test_dataset), cate_correct / len(test_dataset)))
            # print('pi 0: {:.2f}, 1: {:.2f}, 2: {:.2f}'.format(p_max[0], p_max[1], p_max[2]))
            # print('pi 0: {:.2f}, 1: {:.2f}, 2: {:.2f}, 3: {:.2f}, 4: {:.2f},'.format(p_max[0], p_max[1], p_max[2], p_max[3], p_max[4]))
            message = ''
            metric = RetrivalAccMetric(len(test_dataset.labels), 128)
            inshop_acc_in_val, u2shop_acc_in_val, message \
                = eval_retrieval_acc(test_loader, metric, is_cud, model, message, type='Validation')

        # print('Total: {}, Domain Classification Acc: {:.5f}'.format(np.sum(cf_mtx),
        #                                                             (cf_mtx[0] + cf_mtx[3]) / np.sum(cf_mtx)))
        # print('Recall User Photo: {:.5f}'.format(cf_mtx[0] / (cf_mtx[0] + cf_mtx[2])))
        # print('Recall Shop Photo: {:.5f}'.format(cf_mtx[3] / (cf_mtx[1] + cf_mtx[3])))
        print(message)

        with open(os.path.join(save_dir, 'retieval_acc.txt'), 'a') as f:
            # f.write('{},train,inshop,{}\n'.format(model_path.split('/')[-1], ','.join([str(round(a, 4)) for a in inshop_acc_in_train])))
            # f.write('{},train,u2shop,{}\n'.format(model_path.split('/')[-1], ','.join([str(round(a, 4)) for a in u2shop_acc_in_train])))
            f.write('{},val,inshop,{}\n'.format(model_path.split('/')[-1], ','.join([str(round(a, 4)) for a in inshop_acc_in_val])))
            f.write('{},val,u2shop,{}\n'.format(model_path.split('/')[-1], ','.join([str(round(a, 4)) for a in u2shop_acc_in_val])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image retrieval system')
    parser.add_argument('-d', '--dataset', required=True, help='DeepFashion / DeepFashion2')
    # parser.add_argument('-b', '--bbox', required=True, help='gt / pred')
    parser.add_argument('--phase', required=False, default='train', help='Train or Inference')
    parser.add_argument('--cls', dest='domain_cls', required=False, default=False, help='Train or Inference')
    parser.add_argument('-mp', dest='model_path', required=False, default=None, help='pretrained model path')
    parser.add_argument('-s', dest='save_dir', required=True, help='Directory to save the models and the results')
    args = parser.parse_args()

    main(args)
