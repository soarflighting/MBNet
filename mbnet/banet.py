import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from backbone import ResNet
from model_part import PEEModule,MMTLModule,CCFModule,BRModule,FSMModule,_DAHead,AFBNModule

class BANet(nn.Module):
    def __init__(self,in_channels,num_classes,backbone='resnet101',pretrained=True,**kwargs):
        super(BANet, self).__init__()

        self.resnet = ResNet(backbone=backbone,pretrained=pretrained)
        # 因为下采样到1/8 选择output_strides=8
        self.aspp_module = Encoder(output_stride=8)

        self.pee_1 = PEEModule(256)
        self.pee_2 = PEEModule(512)
        self.pee_3 = PEEModule(1024)
        self.pee_4 = PEEModule(2048)

        # self.mmt_1 = MMTLModule(512,4)
        # self.mmt_2 = MMTLModule(512,8)
        # self.mmt_3 = MMTLModule(512,8)
        # self.mmt_4 = MMTLModule(512,8)

        self.br1 = BRModule(512)
        self.br2 = BRModule(512)
        self.br3 = BRModule(512)
        self.br4 = BRModule(512)

        self.ccf_1 = CCFModule(256)
        self.ccf_2 = CCFModule(256)
        self.ccf_3 = CCFModule(256)
        self.ccf_4 = CCFModule(256)

        self.d4 = nn.Conv2d(512, 256, 1, 1)
        self.d3 = nn.Conv2d(512, 256, 1, 1)
        self.d2 = nn.Conv2d(512, 256, 1, 1)
        self.d1 = nn.Conv2d(512, 256, 1, 1)

        ############################################
        self.f1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(256,num_classes,1)
        )
        self.f2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(256,num_classes,1)
        )
        self.f3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(256,num_classes,1)
        )
        ############################################

        self.final = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        imsize = x.size()[2:]
        feature_map,out = self.resnet.base_forward(x)

        aspp_feature = self.aspp_module(feature_map)

        f1p = self.pee_1(feature_map[0])
        f2p = self.pee_2(feature_map[1])
        f3p = self.pee_3(feature_map[2])
        f4p = self.pee_4(feature_map[3])

        f1m = self.br1(f1p)
        f2m = self.br2(f2p)
        f3m = self.br3(f3p)
        f4m = self.br4(f4p)

        # 边界监督模块
        # f1m_1,f1m_2,f1m = self.mmt_1(f1p)
        # f2m_1,f2m_2,f2m = self.mmt_2(f2p)
        # f3m_1,f3m_2,f3m = self.mmt_3(f3p)
        # f4m_1,f4m_2,f4m = self.mmt_4(f4p)

        f1m_ = F.interpolate(f1m,f4m.size()[2:],mode='bilinear',align_corners=True)
        f2m_ = F.interpolate(f2m,f1m.size()[2:],mode='bilinear',align_corners=True)
        f3m_ = F.interpolate(f3m,f1m.size()[2:],mode='bilinear',align_corners=True)
        f4m_ = F.interpolate(f4m,f1m.size()[2:],mode='bilinear',align_corners=True)

        f4c = self.ccf_4(f4m,f1m_,f2m,f3m)
        f3c = self.ccf_3(f3m,f1m_,f2m,f4m)
        f2c = self.ccf_2(f2m,f1m_,f3m,f4m)
        f1c = self.ccf_1(f1m,f2m_,f3m_,f4m_)

        d4 = self.d4(torch.cat([f4c,aspp_feature],dim=1))
        d3 = self.d3(torch.cat([f3c,d4],dim=1))
        d2 = self.d2(torch.cat([f2c,d3],dim=1))

        d1 = self.d1(torch.cat([f1c,F.interpolate(d2,scale_factor=2,mode='bilinear',align_corners=True)],dim=1))

        out1 = F.interpolate(self.f1(d4),imsize,mode='bilinear',align_corners=True)
        out2 = F.interpolate(self.f2(d3), imsize, mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.f3(d2),imsize,mode='bilinear',align_corners=True)
        out = F.interpolate(self.final(d1),imsize,mode='bilinear',align_corners=True)
        if not self.training:
            return out
        return out,out1,out2,out3


if __name__ == '__main__':
    bamodel = BANet(3,2)
    bamodel.train()
    img = torch.rand(2,3,256,256)
    out = bamodel(img)
    print(out[0].size())






