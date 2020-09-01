import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.saved_tensors[0]
        return grad_output.neg() * lambd, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class ResNetbasedNet(nn.Module):
    def __init__(self, load_path=None, depth=101, vec_dim=128, max_pool=False, clf_num=None, adv_eta=None):
        super(ResNetbasedNet, self).__init__()
        self.load = True if load_path is not None else False
        self.clf = True if clf_num is not None else False
        self.adv_eta = Variable(torch.tensor(adv_eta).type(torch.float), requires_grad=False) if adv_eta is not None else None

        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet{}'.format(depth), pretrained=not self.load)
        self.backbone = nn.Sequential(*list(model.children())[:-2])

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1)) if max_pool else nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, vec_dim)
        nn.init.xavier_uniform(self.fc.weight)
        
        if self.clf:
            linear1 = nn.Linear(vec_dim, vec_dim)
            linear2 = nn.Linear(vec_dim, clf_num)
            nn.init.xavier_uniform(linear1.weight)
            nn.init.xavier_uniform(linear2.weight)

            self.clf_layer = nn.Sequential(
                    linear1,
                    nn.BatchNorm1d(vec_dim),
                    nn.ReLU(),
                    linear2)

        if self.load:
            load_model = torch.load(load_path)
            mapped_dict = {'backbone': (self.backbone, {}), 'fc': (self.fc, {})}
            if self.clf:
                mapped_dict['clf_layer'] = (self.clf_layer, {})
            for name, param in load_model.items():
                if name.split('.')[0] in mapped_dict.keys():
                    mapped_dict[name.split('.')[0]][1]['.'.join(name.split('.')[1:])] = param
            for layers in mapped_dict.keys():
                mapped_dict[layers][0].load_state_dict(mapped_dict[layers][1])

    def forward(self, x):
        x = self.backbone(x)
        x = self.max_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x_siam = F.normalize(x, p=2, dim=1)
        if self.clf:
            if self.adv_eta is not None:
                x = grad_reverse(x, self.adv_eta)
            x = self.clf2_layer(x)
            return x_siam, x
        else:
            return x_siam


class MixtureNet(ResNetbasedNet):
    def __init__(self, cfg=None, load_path=None, depth=101, vec_dim=128, max_pool=False,
                 clf1_num=None, clf2_num=None, adv_eta=None, n_comp=3):
        super(MixtureNet, self).__init__(cfg=cfg, load_path=load_path, depth=depth, vec_dim=vec_dim // 2,
                                  max_pool=max_pool, clf1_num=clf1_num, clf2_num=clf2_num, adv_eta=adv_eta)
        self.n_comp = n_comp
        self.mix_pi = nn.Linear(vec_dim // 2, self.n_comp)
        self.mix_emb = nn.Linear(vec_dim // 2, self.n_comp * vec_dim)

        nn.init.xavier_uniform(self.mix_pi.weight)
        nn.init.xavier_uniform(self.mix_emb.weight)

        if load_path is not None:
            load_model = torch.load(load_path)
            mix_pi_map = {p_name.split('.')[1]: load_model[p_name] for p_name in load_model.keys() if p_name.split('.')[0] == 'mix_pi'}
            self.mix_pi.load_state_dict(mix_pi_map)
            mix_emb_map = {p_name.split('.')[1]: load_model[p_name] for p_name in load_model.keys() if p_name.split('.')[0] == 'mix_emb'}
            self.mix_emb.load_state_dict(mix_emb_map)

    def forward(self, x):
        x = self.backbone(x)
        x = self.max_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        pi = F.softmax(self.mix_pi(x), -1)
        x = self.mix_emb(x).reshape(x.shape[0], self.n_comp, -1)
        x = torch.sum(pi.unsqueeze(-1) * x, dim=1)
        x_sim = F.normalize(x, p=2, dim=1)
        if self.clf2:
            if self.adv_eta is not None:
                x = grad_reverse(x, self.adv_eta)
            x2 = self.clf2_layer(x)
            return x_sim, x2, pi
        return x_sim, pi
