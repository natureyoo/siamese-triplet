import numpy as np
from PIL import Image
import cv2
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torchvision.transforms.functional as tf
import albumentations as ab


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, source, n_classes, n_samples):
        self.labels = labels
        self.source = source
        self.labels_set = list(set(self.labels))
        # self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        self.label_to_indices = {label: (i, np.where((self.labels == label) & (self.source == 0))[0],
                                         np.where((self.labels == label) & (self.source == 1))[0])
                                 for i, label in enumerate(self.labels_set)}
        for label in self.labels_set:
            # np.random.shuffle(self.label_to_indices[label])
            np.random.shuffle(self.label_to_indices[label][1])
            np.random.shuffle(self.label_to_indices[label][2])

        self.used_label_indices_count = np.zeros((len(self.labels_set), 2), dtype=int)
        # self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                # indices.extend(self.label_to_indices[class_][
                #                self.used_label_indices_count[class_]:self.used_label_indices_count[
                #                                                          class_] + self.n_samples])
                # self.used_label_indices_count[class_] += self.n_samples
                # if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                #     np.random.shuffle(self.label_to_indices[class_])
                #     self.used_label_indices_count[class_] = 0
                cur_indices = self.label_to_indices[class_]
                if len(cur_indices[1]) > 0 and len(cur_indices[2]) > 0:
                    indices.append(cur_indices[1][self.used_label_indices_count[cur_indices[0], 0]])
                    self.used_label_indices_count[cur_indices[0], 0] += 1
                    cur_count = self.used_label_indices_count[cur_indices[0], 1]
                    indices.extend(cur_indices[2][cur_count:cur_count + self.n_samples - 1])
                    self.used_label_indices_count[cur_indices[0], 1] += self.n_samples - 1

                    if self.used_label_indices_count[cur_indices[0], 0] >= len(cur_indices[1]):
                        np.random.shuffle(self.label_to_indices[class_][1])
                        self.used_label_indices_count[cur_indices[0], 0] = 0

                    if self.used_label_indices_count[cur_indices[0], 1] >= len(cur_indices[2]):
                        np.random.shuffle(self.label_to_indices[class_][2])
                        self.used_label_indices_count[cur_indices[0], 1] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class DABatchSampler(BatchSampler):

    def __init__(self, labels, source, n_classes, n_samples, domain_cls=False):
        self.labels = labels
        self.source = source
        self.labels_set = np.asarray(list(set(self.labels[self.source == 1])))  # treat only shop images' labels
        self.total_label_indices_count = np.zeros(self.labels_set.shape[0], dtype=int)
        self.indices_dict = [np.where(self.source == 0)[0], {}]
        np.random.shuffle(self.indices_dict[0])
        for i, label in enumerate(self.labels_set):
            cur_idx = np.where((self.labels == label) & (self.source == 1))[0]
            self.indices_dict[1][label] = (i, cur_idx)
            np.random.shuffle(self.indices_dict[1][label][1])
            self.total_label_indices_count[i] = len(cur_idx)

        self.used_target_domain_count = 0
        self.used_label_indices_count = np.zeros(len(self.labels_set), dtype=int)
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels[self.source == 1])
        self.batch_size = self.n_samples * self.n_classes

        self.domain_cls = domain_cls

    def __iter__(self):
        self.used_label_indices_count = np.zeros(len(self.labels_set), dtype=int)
        while np.sum(self.used_label_indices_count < self.total_label_indices_count) >= self.n_classes:
            classes = np.random.choice(self.labels_set[self.used_label_indices_count < self.total_label_indices_count],
                                       self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                cur_indices = self.indices_dict[1][class_]
                if len(cur_indices[1]) > 0:
                    cur_count = self.used_label_indices_count[cur_indices[0]]
                    indices.extend(cur_indices[1][cur_count:cur_count + self.n_samples])
                    self.used_label_indices_count[cur_indices[0]] += self.n_samples

                    if self.used_label_indices_count[cur_indices[0]] >= self.total_label_indices_count[cur_indices[0]]:
                        np.random.shuffle(self.indices_dict[1][class_][1])
            if self.domain_cls:
                source_domain_num = len(indices)
                indices.extend(self.indices_dict[0]
                               [self.used_target_domain_count:self.used_target_domain_count + source_domain_num])
                self.used_target_domain_count += source_domain_num
                if self.used_target_domain_count >= len(self.indices_dict[0]):
                    np.random.shuffle(self.indices_dict[0])
                    self.used_target_domain_count = 0

            yield indices

    def __len__(self):
        return self.n_dataset // self.batch_size


class BboxResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bbox):
        x1, y1, x2, y2 = bbox

        return tf.resized_crop(img, y1, x1, y2 - y1, x2 - x1, (self.size, self.size))


class TripletDeepFashion(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class DeepFashionDataset(torch.utils.data.Dataset):
    def __init__(self, ds, root, augment=False, clf=False):
        self.ds = ds
        self.root = root
        self.clf = clf
        self.labels = ds[:, 1]
        self.source = ds[:, 4]
        # self.augment = ab.Compose([
        #     ab.augmentations.transforms.RandomCropNearBBox(max_part_shift=0.3),
        #     ab.OneOf([
        #           ab.HorizontalFlip(p=1),
        #           ab.Rotate(border_mode=1, p=1)
        #     ], p=0.8),
        #     ab.OneOf([
        #           ab.MotionBlur(p=1),
        #           ab.OpticalDistortion(p=1),
        #           ab.GaussNoise(p=1)
        #     ], p=1),
        # ]) if augment else None
        self.augment = ab.HorizontalFlip(p=0.5) if augment else None
        self.transform = ab.Compose([
            ab.Resize(224, 224),
            ab.augmentations.transforms.Normalize(),
        ])

    def __getitem__(self, i):
        # crop bounding box, resize to 224 * 224, normalize
        sample = self.ds[i]
        label = sample[1]
        cate = sample[2]
        source = sample[4]
        image = cv2.imread(os.path.join(self.root, sample[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augment:
            # augmented = self.augment(image=image, cropping_bbox=sample[3])
            bbox = sample[3]
            bbox[2] = min(bbox[2], image.shape[1])
            bbox[3] = min(bbox[3], image.shape[0])
            crop = ab.augmentations.transforms.Crop(bbox[0], bbox[1], bbox[2], bbox[3])
            image = crop(image=image)['image']
            augmented = self.augment(image=image)
            image = augmented['image']
        else:
            bbox = sample[3]
            bbox[2] = min(bbox[2], image.shape[1])
            bbox[3] = min(bbox[3], image.shape[0])
            crop = ab.augmentations.transforms.Crop(bbox[0], bbox[1], bbox[2], bbox[3])
            image = crop(image=image)['image']
        image = self.transform(image=image)['image']
        image = torch.from_numpy(image).permute(2, 1, 0)
        return image, label, cate, source

    def __len__(self):
        return len(self.ds)


class GalleryDataset(torch.utils.data.Dataset):
    def __init__(self, gallery_dir):
        support_img_fmt = ['jpeg', 'jpg', 'jpe', 'png']
        self.list_ids = [file for file in os.listdir(gallery_dir) if file.split('.')[1] in support_img_fmt]
        self.list_imgs = [torch.from_numpy(cv2.imread(os.path.join(gallery_dir, file))).permute(2, 0, 1) for file in self.list_ids]
        self.list_size = [(img.shape[1], img.shape[2]) for img in self.list_imgs]

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        data = {'image': self.list_imgs[index], 'height': self.list_size[index][0], 'width': self.list_size[index][1]}
        return data


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch
