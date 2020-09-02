from itertools import combinations

import numpy as np
import torch
import argparse
import os
import json
from multiprocessing import Process, Manager


def pdist(source_mtx, target_mtx=None):
    if target_mtx is None:
        target_mtx = source_mtx
    distance_matrix = -2 * source_mtx.mm(torch.t(target_mtx))\
                      + (source_mtx ** 2).sum(axis=1).reshape(-1, 1)\
                      + (target_mtx ** 2).sum(axis=1).reshape(1, -1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels, sources):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            anchor = np.where((labels == label) & (sources == 0))[0]
            if len(anchor) < 0:
                continue
            else:
                anchor = anchor.item()
            label_mask = (labels == label)
            positive_indices = np.where(label_mask & (sources == 1))[0]
            if len(positive_indices) < 1:
                continue
            negative_indices = np.where(np.logical_not(label_mask) & (sources == 1))[0]

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor, pos_ind, neg_ind] for pos_ind in positive_indices
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True, domain_adap=False):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.domain_adap = domain_adap
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels, source=None):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            anchor_source = source[label_indices][0] if self.domain_adap else None
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask) & (source != anchor_source))[0] if self.domain_adap \
                                else np.where(np.logical_not(label_mask))[0]
            anchor_positives = [(label_indices[0], i) for i in label_indices[1:]] if self.domain_adap \
                                else list(combinations(label_indices, 2))
            # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False, domain_adap=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu,
                                                                                 domain_adap=domain_adap)


def RandomNegativeTripletSelector(margin, cpu=False, domain_adap=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu,
                                                                                domain_adap=domain_adap)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)


def DomainHardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


# def parse_args_and_merge_const():
#     parser = argparse.ArgumentParser(description='It is the process for Image Retrieval.')
#     parser.add_argument('--conf', default='', type=str)
#     args = parser.parse_args()
#     if args.conf != '':
#         merge_const(args.conf)


def read_data(dataset_name='DeepFashion2', bbox_gt=True, type_list=['train', 'validation']):
    root_path = '/home/jayeon'

    img_list = {}
    item_dict = {}

    if dataset_name == 'DeepFashion':
        file_info = {}
        for idx, line in enumerate(open(os.path.join(root_path, 'Anno/list_bbox_inshop.txt'), 'r').readlines()):
            if idx > 1:     # except first 2 lines
                file_info[line.strip().split()[0]] = np.asarray(line.strip().split()[1:], dtype=np.int)

        # build category idx dictionary
        category_set = set([cate for gender in ['WOMEN', 'MEN'] for cate in os.listdir(os.path.join(base_path, 'img', gender))])
        category_dict = {cate: idx for idx, cate in enumerate(category_set)}

        item_dict = {item.strip(): idx - 1 for idx, item in
                     enumerate(open(os.path.join(root_path, 'Anno/list_item_inshop.txt'), 'r').readlines()) if idx > 0}

        for file_type in type_list:
            img_list[file_type] = []
            is_train = file_type == 'train'
            for idx, line in enumerate(open(os.path.join(root_path, 'Eval/list_eval_partition.txt'), 'r').readlines()):
                if idx > 1 and is_train == (line.strip().split()[2] == 'train'):        # except first 2 lines
                    file_name = line.strip().split()[0]
                    img_list[file_type].append(
                        [file_name, item_dict[file_name.split('/')[-2]], category_dict[file_name.split('/')[2]],
                         file_info[file_name][2:]])

            img_list[file_type] = np.asarray(img_list[file_type], dtype=object)

    elif dataset_name == 'DeepFashion2':
        base_path = os.path.join(root_path, 'DeepFashion2')
        box_key = 'bounding_box' if bbox_gt else 'proposal_boxes'
        for file_type in type_list:
            anno_dir_path = os.path.join(base_path, file_type, 'annos') if bbox_gt \
                else os.path.join('/home/jayeon/TmpData', file_type, 'annos_f')
            img_list[file_type] = []
            item_dict[file_type] = {}
            item_idx = 0
            for file_name in os.listdir(os.path.join(base_path, file_type, 'image')):
                anno_path = os.path.join(anno_dir_path, file_name.split('.')[0] + '.json')
                if not os.path.exists(anno_path):
                    continue
                anno = json.load(open(anno_path, 'r'))
                source_type = 0 if anno['source'] == 'user' else 1
                pair_id = str(anno['pair_id'])
                for key in anno.keys():
                    if key not in ['source', 'pair_id'] and int(anno[key]['style']) > 0:
                        bounding_box = np.asarray(anno[key][box_key], dtype=int)
                        cate_id = anno[key]['category_id'] - 1
                        pair_style = '_'.join([pair_id, str(anno[key]['style'])])
                        if pair_style not in item_dict[file_type].keys():
                            item_dict[file_type][pair_style] = item_idx
                            item_idx += 1
                        img_list[file_type].append([os.path.join(file_type, 'image', file_name),
                                                    item_dict[file_type][pair_style], cate_id,
                                                    bounding_box, source_type])

            img_list[file_type] = np.asarray(img_list[file_type], dtype=object)

    return img_list, base_path, item_dict
