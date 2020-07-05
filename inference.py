import numpy as np
from PIL import Image
import os
from datasets import BboxResizeTransform as br
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from datasets import BalancedBatchSampler, DeepFashionDataset
from networks import ResNet50basedNet
import torch.nn as nn
from torch.utils.data import DataLoader


def find_most_similar(query, gallery_loader, model, cuda, k=10):
    query_img = query[0].float().cuda() if cuda else query[0].float()
    query_vec = model(torch.unsqueeze(query_img, 0))
    top_k_idx = np.zeros(k)
    top_k_correct = np.zeros(k)
    top_k_dist = np.full((k), 1e5)

    for batch_idx, (data, target_label, _, idx) in enumerate(gallery_loader):

        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)

        gallery_vecs = model(*data)[idx != query[3]]
        correct = 1 * (target_label[idx != query[3]] == query[1])  # retrieve or not
        idx = idx[idx != query[3]]
        dist = (query_vec - gallery_vecs).pow(2).sum(dim=1)
        tmp_dist = np.concatenate((top_k_dist, dist.detach().cpu().numpy()))
        tmp_idx = np.concatenate((top_k_idx, idx.numpy()))
        tmp_correct = np.concatenate((top_k_correct, correct))
        top_k_idx = tmp_idx[np.argsort(tmp_dist)[:k]]
        top_k_correct = tmp_correct[np.argsort(tmp_dist)[:k]]
        top_k_dist = np.sort(tmp_dist)[:k]

    return top_k_idx, top_k_correct, top_k_dist


def show_similar_image(query, gallery_loader, model, base_path, cuda, k=10):
    return None


def get_topK_acc(gallery_dataset, item_dict, model, cuda, save_file=None, k=100):
    labels = np.asarray([d[1] for d in gallery_dataset], dtype=int)
    item_to_indices = {label: np.where(labels == label)[0] for label in np.arange(len(item_dict))}
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=4)
    acc = np.zeros(k)
    query_cnt = 0
    for item_idx in item_to_indices.keys():
        if len(item_to_indices[item_idx]) <= 1:
            continue
        if query_cnt % 1 == 0:
            print('finding for {}th query...'.format(query_cnt + 1))
        np.random.shuffle(item_to_indices[item_idx])
        query = gallery_dataset[item_to_indices[item_idx][0]]
        top_k_idx, correct, top_k_dist = find_most_similar(query, gallery_loader, model, cuda, k)
        if save_file is not None:
            save_file.write('{} {} {}'.format(query[3], top_k_idx[:10], top_k_dist[:10]))
        acc += [1 * (np.sum(correct[:k_+1]) > 0) for k_ in range(k)]
        query_cnt += 1
        print(acc / query_cnt)

    acc /= query_cnt

    return acc, query_cnt


def main():
    base_path = '/second/DeepFashion/'
    model_save_path = 'models/siames_triplet.pth'
    backbone = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
    model = ResNet50basedNet(backbone)
    model.load_state_dict(torch.load(model_save_path))

    file_info = {}
    for line in open(os.path.join(base_path, 'Anno/list_bbox_inshop.txt'), 'r').readlines():
        file_name = line.strip().split()[0]
        if file_name.endswith('.jpg'):
            file_info[file_name] = np.asarray(line.strip().split()[1:], dtype=np.int)

    # build category idx dictionary
    category_dict = {}
    category_idx = 0
    for gender in ['WOMEN', 'MEN']:
        for cat in os.listdir(os.path.join(base_path, 'img', gender)):
            if cat not in category_dict.keys():
                category_dict[cat] = category_idx
                category_idx += 1

    item_dict = {item.strip(): idx - 1 for idx, item in enumerate(open(os.path.join(base_path, 'Anno/list_item_inshop.txt'), 'r').readlines()) if idx > 0}

    img_list = {}
    for file_type in ['query', 'gallery']:
        img_list[file_type] = []
        for idx, line in enumerate(open(os.path.join(base_path, 'Eval/list_eval_partition.txt'), 'r').readlines()):
            if idx <= 1:    # except first two lines
                continue
            if file_type == line.strip().split()[2]:
                file_name = line.strip().split()[0]
                img_list[file_type].append([file_name, item_dict[file_name.split('/')[-2]], category_dict[file_name.split('/')[2]], file_info[file_name][2:]])

        img_list[file_type] = np.asarray(img_list[file_type], dtype=object)

    gallery_dataset = DeepFashionDataset(img_list['gallery'], root=base_path)
    gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=64, shuffle=False)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    most_similar_idx = find_most_similar(img_list['query'][0], gallery_loader, model, base_path, cuda, k=10)