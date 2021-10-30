import torch.nn as nn
import torch
import resnet
from torch.nn import functional as F


class ResNet(nn.Module):
    def __init__(self, backbone='resnet101',pretrained=True):
        """Declare all needed layers."""
        super(ResNet, self).__init__()
        if backbone == 'resnet50':
            self.model = resnet.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.model = resnet.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.model = resnet.resnet152(pretrained=pretrained)
        else:
            print("{} model is not exist".format(backbone))
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def base_forward(self, x):

        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out

    def forward(self, x):
        return self.base_forward(x)

if __name__ == '__main__':
    model = ResNet('resnet50',False)

    img = torch.rand(1,3,512,512)
    feature_map,out = model(img)
    print(len(feature_map))
    print(out.size())