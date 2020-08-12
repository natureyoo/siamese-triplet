import numpy as np


def pdist(source_mtx, target_mtx):
    distance_matrix = -2 * source_mtx.dot(target_mtx.transpose()) \
                      + (source_mtx ** 2).sum(axis=1).reshape(-1, 1) \
                      + (target_mtx ** 2).sum(axis=1).reshape(1, -1)
    return distance_matrix


def get_acc(query_emb, query_idx, gall_emb, gall_idx, labels, except_self=False):
    dist = pdist(query_emb, gall_emb)
    if except_self:
        sort_idx = np.argsort(dist, axis=1)[:, 1:21]
    else:
        sort_idx = np.argsort(dist, axis=1)[:, :20]
    match = np.zeros((len(query_idx), 20))
    for i, idx in enumerate(query_idx):
        match[i] = labels[gall_idx[sort_idx[i].astype(np.int)]] == labels[idx]

    acc_val = []
    for k in [1, 5, 10, 20]:
        acc = np.sum(np.sum(match[:, :k], axis=1) > 0) / match.shape[0]
        acc_val.append(acc)

    return acc_val


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'


class RetrivalAccMetric(Metric):
    def __init__(self, data_num, vec_dim=128):
        self.data_num = data_num
        self.vec_dim = vec_dim
        self.emb = np.zeros((self.data_num, self.vec_dim), dtype=np.float16)
        self.label = np.zeros(self.data_num)
        self.source = np.zeros(self.data_num)
        self.cnt = 0

    def __call__(self, outputs, target, source):
        self.emb[self.cnt:self.cnt + outputs.shape[0]] = outputs.detach().cpu().numpy().astype(np.float16)
        self.label[self.cnt:self.cnt + outputs.shape[0]] = target.detach().cpu().numpy()
        self.source[self.cnt:self.cnt + outputs.shape[0]] = source.detach().cpu().numpy()
        self.cnt += outputs.shape[0]

    def reset(self):
        self.emb = np.zeros((self.data_num, self.vec_dim))
        self.label = np.zeros(self.data_num)
        self.source = np.zeros(self.data_num)
        self.cnt = 0

    def value(self):
        user_idx = np.where(self.source == 0)[0]
        shop_idx = np.where(self.source == 1)[0]
        user_emb_mtx = self.emb[user_idx]
        shop_emb_mtx = self.emb[shop_idx]

        inshop_acc = get_acc(shop_emb_mtx, shop_idx, shop_emb_mtx, shop_idx, self.label, True)
        u2shop_acc = get_acc(user_emb_mtx, user_idx, shop_emb_mtx, shop_idx, self.label)

        return inshop_acc, u2shop_acc

    def name(self):
        return 'Retrieval Accuracy'
