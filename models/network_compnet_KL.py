import torch
import torch.nn as nn
import models.basicblock as B
# 还是不好的话，可以去掉那两次注意力，把nc改成128
class D_Block(nn.Module):
    def __init__(self, channel_in, channel_out, deconv = False):
        super(D_Block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, stride=1,
                                padding=1)
        self.relu1 = nn.PReLU()
        # if deconv: # down层采用deconv，up层采用conv
        #     self.conv_2 = nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.),
        #                             kernel_size=3, stride=1, padding=2, bias=True, dilation=2)
        # 还是都卷积吧
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
# change pool to conv
class _down(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(_down, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=4, stride=2, padding=1)

        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))

        return out
# PixelShuffle + nolocal + 1x1 conv
class _up(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True, upFactor=2):
        super().__init__()
        assert in_channels%4 == 0
        self.up = nn.PixelShuffle(upscale_factor=upFactor)
        self.nolocal = B.NonLocalBlock_NLRN(int(in_channels/4), 4)
        self.conv2 =B.conv(in_channels=int(in_channels/4), out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias,mode='CR')

    def forward(self,x):
        out = self.up(x)
        out = self.nolocal(out)
        out = self.conv2(out)
        return out

class Net(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        super(Net, self).__init__()
        self.channel_trans = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=True)

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=True)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=True) for _ in range(7)]
        self.image_net = B.sequential(m_head, *m_body)
        self.image_net_tail = B.conv(nc, out_nc, mode='C', bias=True)

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=True)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=True) for _ in range(7)]
        self.noise_net = B.sequential(m_head, *m_body)
        self.noise_net_tail = B.conv(nc, out_nc, mode='C', bias=True)
        #　get mean and var
        # self.noise_net_tail1 = nn.Sequential(
        #     B.conv(nc, nc, mode='C', bias=True),
        #     nn.AdaptiveAvgPool2d((10, 10)),
        # )
        # self.noise_net_tail = nn.Sequential(
        #     nn.Linear(nc*10*10,2),
        #     nn.Softmax()
        # )

        m_body = [B.conv(nc*2, nc*2, mode='C'+act_mode, bias=True) for _ in range(3)]
        m_tail = B.conv(nc*2, out_nc, mode='C', bias=True)
        self.comp = B.sequential(*m_body, m_tail)

    def forward(self, x, train=True): # 非residual
        if train:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x1_ct = self.channel_trans(x1)
            image1 = self.image_net(x1)
            image2 = self.image_net_tail(image1)

            noise1_1 = self.noise_net(x1)
            noise1_2 = self.noise_net_tail(noise1_1)

            noise2_1 = self.noise_net(x2)
            noise2_2 = self.noise_net_tail(noise2_1)

            noise3_1 = self.noise_net(x3)
            noise3_2 = self.noise_net_tail(noise3_1)

            noise4_1 = self.noise_net(x4)
            noise4_2 = self.noise_net_tail(noise4_1)

            noise1 = x1_ct - image1
            noise = torch.cat([noise1,noise1_1],1)
            noise = self.comp(noise)
            out = x1 - noise

            return out, image2, [noise1_2, noise2_2, noise3_2, noise4_2]

        else:
            image1 = self.image_net(x)
            # image2 = self.image_net_tail(image1)
            noise1_1 = self.noise_net(x)
            # noise1_2 = self.noise_net_tail(noise1_1)
            x_ct = self.channel_trans(x)
            noise1 = x_ct - image1
            noise = torch.cat([noise1,noise1_1],1)
            noise = self.comp(noise)
            out = x - noise
            return out
