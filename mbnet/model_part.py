import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SENet(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PEEModule(nn.Module):

    def __init__(self,in_channels):
        super(PEEModule, self).__init__()
        self.ext = nn.Conv2d(in_channels,128,kernel_size=1)

        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(3)
        self.avg3 = nn.AdaptiveAvgPool2d(6)
        self.avg4 = nn.AdaptiveAvgPool2d(8)

    def forward(self, x):
        imsize = x.size()[2:]

        feat = self.ext(x)
        avg_1 = self.avg1(feat)
        avg_2 = self.avg2(feat)
        avg_3 = self.avg3(feat)
        avg_4 = self.avg4(feat)

        avg_1 = F.interpolate(avg_1,imsize,mode="bilinear",align_corners=True)
        avg_2 = F.interpolate(avg_2,imsize,mode='bilinear',align_corners=True)
        avg_3 = F.interpolate(avg_3,imsize,mode='bilinear',align_corners=True)
        avg_4 = F.interpolate(avg_4,imsize,mode='bilinear',align_corners=True)

        feat1 = feat - avg_1
        feat2 = feat - avg_2
        feat3 = feat - avg_3
        feat4 = feat - avg_4

        feat_c = torch.cat([feat1,feat2,feat3,feat4],dim=1)

        return feat_c

class MMTLModule(nn.Module):

    def __init__(self,in_channels,up_num):
        super(MMTLModule, self).__init__()

        self.up = up_num
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//4,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(True)
        )

        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//4,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU()
        )

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels//4,in_channels//2,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU()
        )

        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels//4,in_channels//2,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU()
        )

        self.conv_edge = nn.Sequential(
            nn.Conv2d(in_channels//2,1,1),
            nn.Dropout()
        )
        self.conv_seg = nn.Sequential(
            nn.Conv2d(in_channels//2,1,1),
            nn.Dropout()
        )

        self.sigmoid = nn.Sigmoid()

        self.final = nn.Sequential(
            nn.Conv2d(in_channels,256,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )


    def forward(self, x):

        # 第一个卷积层
        feat_e_1 = self.conv_1_1(x)
        feat_s_1 = self.conv_1_2(x)

        # 交互模块中的fe
        feat_sigmoid_e = self.sigmoid(feat_e_1)
        feat_sigmoid_e = 1 - feat_sigmoid_e
        feat_fusion_e = torch.mul(feat_sigmoid_e,feat_s_1)
        feat_e = torch.add(feat_fusion_e,feat_e_1)

        # 交互模块中fs
        feat_sigmoid_s = self.sigmoid(feat_s_1)
        feat_sigmoid_s = 1 - feat_sigmoid_s
        feat_fusion_s = torch.mul(feat_sigmoid_s,feat_e_1)
        feat_s = torch.add(feat_fusion_s,feat_s_1)

        feat_e_2 = self.conv_2_1(feat_e)
        feat_s_2 = self.conv_2_2(feat_s)

        feat_m = torch.cat([feat_e_2,feat_s_2],dim=1)

        # 边界预测
        feat1 = F.interpolate(feat_e_2,scale_factor=self.up,mode='bilinear',align_corners=True)
        feat1 = self.conv_edge(feat1)

        # 掩码图预测
        feat2 = F.interpolate(feat_s_2,scale_factor=self.up,mode='bilinear',align_corners=True)
        feat2 = self.conv_seg(feat2)

        feat_m = self.final(feat_m)

        return feat1,feat2,feat_m


## 边界细化模块
class BRModule(nn.Module):
    def __init__(self,in_channels,channels=256):
        super(BRModule, self).__init__()

        self.reduction = nn.Conv2d(in_channels,channels,1)

        self.br1 = nn.Sequential(nn.Conv2d(channels,channels//2,kernel_size=3,padding=1),
                                 nn.BatchNorm2d(channels//2),
                                 nn.ReLU(True))
        self.br2 = nn.Sequential(nn.Conv2d(channels//2,channels,kernel_size=3,padding=1),nn.BatchNorm2d(channels))

        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.reduction(x)
        residual = x
        out = self.br1(x)
        out = self.br2(out)
        out = self.relu(residual+out)
        return out

# 大小和维度都不改变
class CCFModule(nn.Module):
    def __init__(self,channel):
        super(CCFModule, self).__init__()
        self.sigmoid = nn.Sigmoid()

        # senet
        self.senet = SENet(channel=channel)
        self.c = nn.Conv2d(channel,channel,1)

    def forward(self, x1,x2,x3,x4):
        feat1 = self.sigmoid(x1)
        # 改变处
        # feat1 = 1 - feat1

        feat2 = self.sigmoid(x2)
        feat2 = torch.mul(feat2,x2)
        feat3 = self.sigmoid(x3)
        feat3 = torch.mul(feat3,x3)
        feat4 = self.sigmoid(x4)
        feat4 = torch.mul(feat4,x4)

        feat234 = torch.add(feat2,torch.add(feat3,feat4))

        feat = torch.mul(feat234,feat1)

        feat_c = torch.add(x1,feat)

        feat = self.c(self.senet(feat_c))

        return feat


class FSMModule(nn.Module):
    def __init__(self,in_channels,channels):

        super(FSMModule, self).__init__()
        self.reduction = nn.Conv2d(in_channels,channels,3,padding=1,stride=1)

        self.inter_channel = channels // 2
        self.conv_phi = nn.Conv2d(in_channels=channels, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channels, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channels, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_reduction = nn.Conv2d(channels,in_channels,kernel_size=3,padding=1)

    def forward(self,x):

        rediual = x

        x = self.reduction(x)
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x

        out = self.conv_reduction(out)

        out += rediual
        return out



class PSPModule(nn.Module):
    def __init__(self,sizes = (1,3,6,8),dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size,dimension) for size in sizes])

    def _make_stage(self,size,dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self,feats):
        n,c,_,_ = feats.size()
        priors = [stage(feats).view(n,c,-1) for stage in self.stages]
        center = torch.cat(priors,-1)
        return center


class _SelfAttentionBlock(nn.Module):
    def __init__(self,low_in_channels,high_in_channels,key_channels,value_channels,out_channels=None, psp_size=(1,3,6,8)):
        super(_SelfAttentionBlock, self).__init__()
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channels,out_channels=key_channels,kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(True)
        )

        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channels,out_channels=key_channels,kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(True)
        )

        self.f_value = nn.Conv2d(in_channels=low_in_channels,out_channels=value_channels,kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=value_channels,out_channels=out_channels,kernel_size=1, stride=1, padding=0)
        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight,0)
        nn.init.constant_(self.W.bias,0)

    def forward(self,low_feats,high_feats):
        batch_size,c,h,w = high_feats.size()
        value = self.psp(self.f_value(low_feats))

        query = self.f_query(high_feats).view(batch_size,self.key_channels,-1)
        query = query.permute(0,2,1)
        key = self.f_key(low_feats)
        value = value.permute(0,2,1)
        key = self.psp(key)

        sim_map = torch.matmul(query,key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map,value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *high_feats.size()[2:])
        context = self.W(context)
        return context


class AFBNModule(nn.Module):
    def __init__(self,low_in_channles,high_in_channels,inter_channels,out_channels):
        super(AFBNModule, self).__init__()

        self.selfAttention = _SelfAttentionBlock(low_in_channels=low_in_channles,high_in_channels=high_in_channels,
                                                 key_channels=inter_channels,value_channels=inter_channels,out_channels=out_channels)


    def forward(self,low_feats,high_feats):
        context = self.selfAttention(low_feats,high_feats)
        return context

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, channels, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        print("danet...")
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.6),
            nn.Conv2d(inter_channels, channels, 1)
        )
        # if aux:
        #     self.conv_p3 = nn.Sequential(
        #         nn.Dropout(0.5),
        #         nn.Conv2d(inter_channels, nclass, 1)
        #     )
        #     self.conv_c3 = nn.Sequential(
        #         nn.Dropout(0.5),
        #         nn.Conv2d(inter_channels, nclass, 1)
        #     )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        # if self.aux:
        #     p_out = self.conv_p3(feat_p)
        #     c_out = self.conv_c3(feat_c)
        #     outputs.append(p_out)
        #     outputs.append(c_out)

        # return tuple(outputs)
        return fusion_out



if __name__ == '__main__':
    # Cmodel = CCFModule()
    # image_infer = torch.rand(2,64,256,256)
    # out = Cmodel(image_infer,image_infer,image_infer,image_infer)
    # print(out.size())
    # image_infer = torch.rand(2,256,128,128)
    # pee = PEEModule(256)
    # out = pee(image_infer)
    # print(out.size())

    model =  _DAHead(2048,256)
    img = torch.rand(2,2048,32,32)
    out = model(img)
    print(out.size())



