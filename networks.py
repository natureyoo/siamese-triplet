import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.layers import ShapeSpec
import fvcore.nn.weight_init as weight_init


class ResNetbasedNet(nn.Module):
    def __init__(self, cfg):
        super(ResNetbasedNet, self).__init__()
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

        self.backbone = nn.Sequential(*list(model.children()))    # conv layer of ResNet50
        self.max_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 128)
        weight_init.c2_xavier_fill(self.fc)

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
