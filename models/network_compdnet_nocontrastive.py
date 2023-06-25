import torch
import torch.nn as nn
import models.basicblock as B
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 还是不好的话，可以去掉那两次注意力，把nc改成128
class D_Block(nn.Module):
    def __init__(self, channel_in, channel_out, deconv = False):
        super(D_Block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, stride=1,
                                padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,
                                stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1,
                                padding=1)
        self.relu3 = nn.PReLU()
        self.tail = B.conv(channel_in, channel_out, mode='CBR')
    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)
        out = self.tail(out)
        return out
class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()

        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)

        out = self.relu(self.conv(out))

        return out
class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()

        self.relu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv(x))

        out = self.subpixel(out)

        return out
class UNet_org(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(UNet_org, self).__init__()
        self.DCR_block11 = D_Block(in_nc, in_nc)
        self.DCR_block12 = D_Block(in_nc, in_nc)
        self.down1 = self.make_layer(_down, in_nc)
        self.DCR_block21 = D_Block(in_nc*2, in_nc*2)
        self.DCR_block22 = D_Block(in_nc*2, in_nc*2)
        self.down2 = self.make_layer(_down, in_nc*2)
        self.DCR_block31 = D_Block(in_nc*4, in_nc*4)
        self.DCR_block32 = D_Block(in_nc*4, in_nc*4)
        self.down3 = self.make_layer(_down, in_nc*4)
        self.DCR_block41 = D_Block(in_nc*8, in_nc*8)
        self.DCR_block42 = D_Block(in_nc*8, in_nc*8)
        self.up3 = self.make_layer(_up, in_nc*16)
        self.DCR_block33 = D_Block(in_nc*8, in_nc*8)
        self.DCR_block34 = D_Block(in_nc*8, in_nc*8)
        self.up2 = self.make_layer(_up, in_nc*8)
        self.DCR_block23 = D_Block(in_nc*4, in_nc*4)
        self.DCR_block24 = D_Block(in_nc*4, in_nc*4)
        self.up1 = self.make_layer(_up, in_nc*4)
        self.DCR_block13 = D_Block(in_nc*2, in_nc*2)
        self.DCR_block14 = D_Block(in_nc*2, out_nc)
        # self.conv_f = nn.Conv2d(in_channels=in_nc*2, out_channels=out_nc, kernel_size=1, stride=1, padding=0)
        # self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.DCR_block11(x)

        conc1 = self.DCR_block12(out)

        out = self.down1(conc1)

        out = self.DCR_block21(out)

        conc2 = self.DCR_block22(out)

        out = self.down2(conc2)

        out = self.DCR_block31(out)

        conc3 = self.DCR_block32(out)

        conc4 = self.down3(conc3)

        out = self.DCR_block41(conc4)

        out = self.DCR_block42(out)

        out = torch.cat([conc4, out], 1)

        out = self.up3(out)

        out = torch.cat([conc3, out], 1)

        out = self.DCR_block33(out)

        out = self.DCR_block34(out)

        out = self.up2(out)

        out = torch.cat([conc2, out], 1)

        out = self.DCR_block23(out)

        out = self.DCR_block24(out)

        out = self.up1(out)

        out = torch.cat([conc1, out], 1)

        out = self.DCR_block13(out)

        out = self.DCR_block14(out)

        # out = self.relu2(self.conv_f(out))

        return out

class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1
class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        return d1
class Net(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR', nG=5): # nG为混合高斯个数
        super(Net, self).__init__()
        self.channel_trans = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=True)

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=True)
        m_body = [D_Block(nc, nc) for _ in range(7)]
        # m_body_ = Recurrent_block(nc)
        self.image_net = B.sequential(m_head, *m_body)
        self.image_net_tail = B.conv(nc, out_nc, mode='C', bias=True)

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=True)
        m_body = [D_Block(nc, nc) for _ in range(7)]
        # m_body_ = Recurrent_block(nc)
        self.noise_net = B.sequential(m_head, *m_body)
        #　get mean and var
        m_body_ = B.conv(nc*2, nc, mode='C'+act_mode[-1], bias=True)
        m_body = [UNet_org(nc, nc)  for _ in range(1)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=True)
        self.comp = B.sequential(m_body_, *m_body, m_tail)

    def forward(self, x, train=True): # 非residual
        if train:
            x1 = x[0]
            x1_ct = self.channel_trans(x1)
            image1 = self.image_net(x1)
            image2 = self.image_net_tail(image1)

            noise1 = self.noise_net(x1)

            noise = x1_ct - image1
            noise_ = torch.cat([noise,noise1],1)
            noise_ = self.comp(noise_)
            out = x1 - noise_

            return out, image2

        else:
            image1 = self.image_net(x)
            noise1 = self.noise_net(x)
            x_ct = self.channel_trans(x)
            noise = x_ct - image1
            noise = torch.cat([noise,noise1],1)
            noise = self.comp(noise)
            out = x - noise
            return out
