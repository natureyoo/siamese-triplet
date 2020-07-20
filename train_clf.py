import numpy as np
import argparse, time, os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim

from datasets import BalancedBatchSampler, DeepFashionDataset
from networks import ResNetbasedNet
from losses import OnlineTripletLoss
from utils import read_data, pdist, RandomNegativeTripletSelector, HardestNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from trainer_clf import fit


def main(args):
    model_save_dir = args.save_dir
    vec_dim = 512

    data_type = ['validation'] if args.phase == 'test' else ['train', 'validation']
    img_list, base_path, item_dict = read_data("DeepFashion2", bbox_gt=True, type_list=data_type)
    clf_inst_num = len(set(img_list['train'][:, 1]))
    clf_cate_num = len(set(img_list['train'][:, 2]))

    model = ResNetbasedNet(vec_dim=vec_dim, max_pool=True, clf1_num=clf_inst_num, clf2_num=clf_cate_num)
    # model.load_state_dict(torch.load('/home/jayeon/Documents/siamese-triplet/model_clf/00006.pth'))
    model.load_state_dict(torch.load(args.model_path))
    # model.fc = nn.Linear(2048, vec_dim)
    # model.load_state_dict(torch.load('./model_adap_hardes_continue_norm69/00005.pth'))
    # model.load_state_dict(torch.load(args.model_path))

    # domain_adap = args.domain_adap
    is_cud = torch.cuda.is_available()
    device = torch.device("cuda" if is_cud else "cpu")
    if is_cud:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    kwargs = {'num_workers': 8, 'pin_memory': True} if is_cud else {}

    if args.phase == 'train':
        train_dataset = DeepFashionDataset(img_list['train'], root=base_path, augment=True, clf=True)
        online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, **kwargs)

        test_dataset = DeepFashionDataset(img_list['validation'], root=base_path, clf=True)
        online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, **kwargs)

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()
        grad_norm = 1.
        optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)
        nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        scheduler = lr_scheduler.StepLR(optimizer, 4, gamma=0.9, last_epoch=-1)
        n_epochs = 500
        log_interval = 200

        fit(online_train_loader, online_test_loader, model, criterion1, criterion2, optimizer, scheduler, n_epochs,
            is_cud, log_interval, model_save_dir, start_epoch=87)

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
            for idx, (data, target, _) in enumerate(test_loader):
                emb_vecs = model(data.cuda())
                embedding_mtx[idx_: idx_ + len(data)] = emb_vecs
                labels[idx_:idx_ + len(data)] = np.asarray(target)
                idx_ += len(data)
                if idx % 20 == 0:
                    print(
                        'processing {}/{}... elapsed time {}s'.format(idx + 1, len(test_loader), time.time() - start_time))
        np.save(args.result_path, embedding_mtx)
        with open(os.path.join('/'.join(args.result_path.split('/')[:-1]), 'file_info.txt'), 'w') as f:
            for i in range(len(test_dataset)):
                f.write('{},{},{}\n'.format(img_list['validation'][i][0], test_dataset[i][1], test_dataset[i][2]))
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
    parser.add_argument('--domain', dest='domain_adap', required=False, default=False, help='Train or Inference')
    parser.add_argument('-mp', '--model_path', dest='model_path',  required=True)
    parser.add_argument('-s', dest='save_dir', required=False, help='Directory to save the results')
    args = parser.parse_args()

    main(args)
