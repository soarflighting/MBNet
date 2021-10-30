import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from backbone import ResNet


class Dran(ResNet):
    def __init__(self,n_classes,aux=False,backbone='resnet101',pretrained=True,**kwargs):
        super(Dran, self).__init__()
        self.head = DranHead(2048,n_classes)
        in_channels = 256
        self.aux = aux
        if aux:
            self.cls_aux = nn.Sequential(
                nn.Conv2d(1024,in_channels,3,padding=1,bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Dropout2d(0.1,False),
                nn.Conv2d(in_channels,n_classes,1)
            )
        self.cls_seg = nn.Sequential(
            nn.Dropout2d(0.5,False),
            nn.Conv2d(in_channels,n_classes,1)
        )

    def forward(self, x):
        imsize = x.size()[2:]
        multix,out = self.base_forward(x)

        ## dran head for seg
        final_feat = self.head(multix)
        cls_seg = self.cls_seg(final_feat)
        cls_seg = F.interpolate(cls_seg,imsize,mode="bilinear",align_corners=True)

        ## aux head for seg
        outputs = [cls_seg]
        if self.aux:
            cls_aux = self.cls_aux(multix[-2])
            cls_aux = F.interpolate(cls_aux,imsize,mode='bilinear',align_corners=True)
            outputs.append(cls_aux)

        return tuple(outputs)




class DranHead(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DranHead, self).__init__()
        inter_channels = in_channels//4

        #Convs or modules for CPAM
        self.conv_cpam_b = nn.Sequential(
            nn.Conv2d(in_channels,inter_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )

        self.cpam_enc = CPAMEnc(inter_channels)
        self.cpam_dec = CPAMDec(inter_channels)
        self.conv_cpam_e = nn.Sequential(
            nn.Conv2d(inter_channels,inter_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )

        # Convs or modules for CCAM
        self.conv_ccam_b = nn.Sequential(
            nn.Conv2d(in_channels,inter_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )

        self.ccam_enc = nn.Sequential(
            nn.Conv2d(inter_channels,inter_channels//16,1,bias=False),
            nn.BatchNorm2d(inter_channels//16),
            nn.ReLU(True)
        )

        self.ccam_dec = CCAMDec()
        self.conv_ccam_e = nn.Sequential(
            nn.Conv2d(inter_channels,inter_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )
        # fusion conv
        self.conv_cat = nn.Sequential(
            nn.Conv2d(inter_channels*2,inter_channels//2,3,padding=1,bias=False),
            nn.BatchNorm2d(inter_channels//2),
            nn.ReLU()
        )

        ## Cross-level Gating Decoder(CLGD)
        self.clgd = CLGD(inter_channels//2,inter_channels//2)

    def forward(self, multix):

        ## Compact Channel Attention Module(CCAM)
        ccam_b = self.conv_ccam_b(multix[-1])
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b,ccam_f)

        ## Compact Spatial Attention Module(CPAM)
        cpam_b = self.conv_cpam_b(multix[-1])
        cpam_f = self.cpam_enc(cpam_b).permute(0,2,1)#BKD
        cpam_feat = self.cpam_dec(cpam_b,cpam_f)

        ## Fuse two modules
        ccam_feat = self.conv_ccam_e(ccam_feat)
        cpam_feat = self.conv_cpam_e(cpam_feat)
        feat_sum = self.conv_cat(torch.cat([cpam_feat,ccam_feat],1))

        ## Cross-level Gating Decoder(CLGD)
        final_feat = self.clgd(multix[0], feat_sum)

        return final_feat

class CPAMEnc(nn.Module):
    '''
    CPAM encoding module
    '''
    def __init__(self,in_channels):
        super(CPAMEnc, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        b,c,h,w = x.size()

        feat1 = self.conv1(self.pool1(x)).view(b,c,-1)
        feat2 = self.conv2(self.pool2(x)).view(b,c,-1)
        feat3 = self.conv3(self.pool3(x)).view(b,c,-1)
        feat4 = self.conv4(self.pool4(x)).view(b,c,-1)

        return torch.cat((feat1,feat2,feat3,feat4),2)


class CPAMDec(nn.Module):
    '''
    CPAM decoding module
    '''
    def __init__(self,in_channels):
        super(CPAMDec, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

        self.conv_query = nn.Conv2d(in_channels,in_channels//4,kernel_size=1)
        self.conv_key = nn.Linear(in_channels,in_channels//4)
        self.conv_value = nn.Linear(in_channels,in_channels)

    def forward(self, x,y):

        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batch_size,C,width,height = x.size()
        m_batch_size,K,M = y.size()

        proj_query = self.conv_query(x).view(m_batch_size,-1,width*height).permute(0,2,1)#BXNXd
        proj_key = self.conv_key(y).view(m_batch_size,K,-1).permute(0,2,1)#BxdXk
        energy = torch.bmm(proj_query,proj_key) #BXNXK
        attention = self.softmax(energy) #BXNXK

        proj_value = self.conv_value(y).permute(0,2,1) #BXCXK
        out = torch.bmm(proj_value,attention.permute(0,2,1)) #BXCXN
        out = out.view(m_batch_size,C,width,height)
        out = self.scale*out + x

        return out


class CCAMDec(nn.Module):
    '''
    CCAM decoding module
    '''
    def __init__(self):
        super(CCAMDec, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x,y):
        m_batchsize,C,width ,height = x.size()
        x_reshape =x.view(m_batchsize,C,-1)

        B,K,W,H = y.size()
        y_reshape =y.view(B,K,-1)
        proj_query  = x_reshape #BXC1XN
        proj_key  = y_reshape.permute(0,2,1) #BX(N)XC
        energy =  torch.bmm(proj_query,proj_key) #BXC1XC
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B,K,-1) #BCN

        out = torch.bmm(attention,proj_value) #BC1N
        out = out.view(m_batchsize,C,width ,height)

        out = x + self.scale*out
        return out


class CLGD(nn.Module):
    '''
    Cross-level Gating Decoder
    '''
    def __init__(self,in_channels,out_channels):
        super(CLGD, self).__init__()

        inter_channels = 32
        self.conv_low = nn.Sequential(
            nn.Conv2d(in_channels,inter_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels,in_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        self.conv_att = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels,1,1),
            nn.Sigmoid()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x,y):
        """
            inputs :
                x : low level feature(N,C,H,W)  y:high level feature(N,C,H,W)
            returns :
                out :  cross-level gating decoder feature
        """
        low_lvl_feat = self.conv_low(x)
        high_lvl_feat = F.interpolate(y,low_lvl_feat.size()[2:],mode='bilinear',align_corners=True)
        feat_cat = torch.cat([low_lvl_feat,high_lvl_feat],1)

        low_lvl_feat_refine = self.gamma*self.conv_att(feat_cat)*low_lvl_feat
        low_high_feat = torch.cat([low_lvl_feat_refine,high_lvl_feat],1)
        low_high_feat = self.conv_cat(low_high_feat)

        low_high_feat = self.conv_out(low_high_feat)

        return low_high_feat



if __name__ == '__main__':
    model = Dran(n_classes=2,pretrained=False)
    img = torch.rand(2,3,512,512)
    out = model(img)
    print(out[0].size())








