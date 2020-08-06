import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.layers import ShapeSpec
import fvcore.nn.weight_init as weight_init


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
    def __init__(self, cfg=None, load_path=None, depth=101, vec_dim=128, max_pool=False, clf1_num=None, clf2_num=None, adv_eta=None):
        super(ResNetbasedNet, self).__init__()
        self.load = True if load_path is not None else False
        self.clf1 = True if clf1_num is not None else False
        self.clf2 = True if clf2_num is not None else False
        self.adv_eta = Variable(torch.tensor(adv_eta).type(torch.float), requires_grad=False) if adv_eta is not None else None

        if cfg is not None:
            model = build_resnet_backbone(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
            pretrained_model = torch.load(cfg.MODEL.WEIGHTS)
            cur_state = model.state_dict()
            mapped_dict = {}
            for name, param in pretrained_model.items():
                if name == 'model':
                    for p in param:
                        if p.replace('backbone.bottom_up.', '') in cur_state:
                            mapped_dict[p.replace('backbone.bottom_up.', '')] = param[p]
            model.load_state_dict(mapped_dict)
            self.backbone = nn.Sequential(*list(model.children()))
        else:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet{}'.format(depth), pretrained=not self.load)
            self.backbone = nn.Sequential(*list(model.children())[:-2])

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1)) if max_pool else nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, vec_dim)
        
        if self.clf1:
            self.clf1_layer = nn.Sequential(
                    nn.Linear(vec_dim, vec_dim),
                    nn.BatchNorm1d(vec_dim),
                    nn.ReLU(),
                    nn.Linear(vec_dim, clf1_num))

        if self.clf2:
            self.clf2_layer = nn.Sequential(
                    nn.Linear(vec_dim, vec_dim),
                    nn.BatchNorm1d(vec_dim),
                    nn.ReLU(),
                    nn.Linear(vec_dim, clf2_num))

        if self.load:
            load_model = torch.load(load_path)
            mapped_dict = {'backbone':(self.backbone, {}), 'fc':(self.fc, {})}
            if self.clf1:
                mapped_dict['clf1_layer'] = (self.clf1_layer, {})
            if self.clf2:
                # print(self.clf2_layer.state_dict())
                mapped_dict['clf2_layer'] = (self.clf2_layer, {})
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
        if not self.clf1 or not self.clf2:
            x_siam = F.normalize(x, p=2, dim=1)
            if self.clf2:
                x2 = self.clf2_layer(x)
                if self.adv_eta is not None:
                    x2 = grad_reverse(x2, self.adv_eta)
                return x_siam, x2
            return x_siam
        else:
            x1 = self.clf1_layer(x)
            x2 = self.clf2_layer(x)
            return x1, x2


class ResNet50basedNet(nn.Module):
    def __init__(self, backbone_model):
        super(ResNet50basedNet, self).__init__()
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])    # conv layer of ResNet50
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(2048, 256)

    def forward(self, x):
        x = self.backbone(x)
        x = self.max_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class ResNet50basedMultiNet(nn.Module):
    def __init__(self):
        super(ResNet50basedMultiNet, self).__init__()

    def forward(self, x):
        x = self.fc(x)
        return x


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
