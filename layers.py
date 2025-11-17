## -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.quantization import QuantStub, DeQuantStub
from cbam import CBAM

# theta+Omega
class Conv_RFF(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=0, mc=10, kernel_type='RBF',
                 group=1, F0=True):
        super(Conv_RFF, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels // group
        self.kernel_size = k_size
        self.stride = stride
        self.padding = padding
        self.mc = mc
        self.kernel_type = kernel_type
        self.group = group
        self.F0 = F0

        # theta:scaler
        self.theta_logsigma = nn.Parameter(torch.zeros(1), requires_grad=True)
        # scaler
        self.llscale = nn.Parameter(0.5 * torch.log(torch.tensor(self.in_channels * k_size * k_size).float()),
                                    requires_grad=False)
        # [C*k*k,C']
        self.theta_llscale = nn.Parameter(
            torch.ones([self.in_channels * k_size * k_size]) * self.llscale, requires_grad=True)
        # Omega:q
        # Omega:[C*k*k,C']
        self.Omega_mean = nn.Parameter(torch.zeros(self.in_channels * k_size * k_size, self.out_channels),
                                       requires_grad=True)
        # Omega:[C*k*k,C']
        self.Omega_logsigma = nn.Parameter(
            (self.theta_llscale * -2.)[..., None] * torch.ones(self.in_channels * k_size * k_size, self.out_channels),
            requires_grad=True)
        # Omega_eps:[mc,C*k*k,C']
        self.Omega_eps = nn.Parameter(
            torch.randn(self.mc, self.in_channels * self.kernel_size * self.kernel_size, self.out_channels),
            requires_grad=True)

    def get_prior_Omega(self):
        # [C*k*k,1]
        Omega_mean_prior = torch.zeros(self.in_channels * self.kernel_size * self.kernel_size, 1)
        # [C*k*k]
        Omega_logsigma_prior = self.theta_llscale * -2. * torch.ones(
            self.in_channels * self.kernel_size * self.kernel_size).cuda()
        return Omega_mean_prior, Omega_logsigma_prior

    def forward(self, x):
        # input:
        # original image:[batch_size,C,H,W]->[8,1,28,28]
        # or the output feature map of last layer:[batch_size,mc*channels,H,W]
        if self.F0:
            x = torch.unsqueeze(x, 0).repeat(self.mc, 1, 1, 1, 1)
            x = x.transpose(0, 1)  # 5d

        # local reparameter
        # self.Omega_from_q = Omega_eps * torch.exp(self.Omega_logsigma / 2) + self.Omega_mean
        self.Omega_from_q = self.Omega_eps * torch.exp(self.Omega_logsigma / 2) + self.Omega_mean

        # conv2d_rff
        # x:[batch_size,mc*channels,H,W]
        # weight:self.Omega_from_q [mc,C*k*k,C']
        # output:[batch,mc*channels,H,W]
        # new: r_n, mc, out_channel, r_h, r_w
        phi_half = conv2d(x, self.Omega_from_q, self.in_channels, self.out_channels, self.kernel_size, self.stride,
                          self.padding, self.group)
        ## RFF:[cos(F*x),sin(F*x)]
        phi_half_size = phi_half.shape[-1]
        N_rf = self.out_channels * phi_half_size * phi_half_size
        phi = torch.exp(0.5 * self.theta_logsigma)
        phi = phi / torch.sqrt(torch.tensor(1.) * N_rf)

        # phi_half = phi_half.view(-1, self.mc, self.out_channels, phi_half_size, phi_half_size)
        if self.kernel_type == "RBF":
            A = phi * torch.cos(phi_half)
            B = phi * torch.sin(phi_half)
            # output:[batch,mc*channels*2,H,W]
            phi = torch.cat([A, B], 2)
        else:
            # phi = phi * torch.cat([torch.maximum(phi_half, torch.tensor(0.))], 2)
            phi = phi * torch.cat([torch.relu(phi_half)], 2)
        # phi = phi.view(-1, self.mc * phi.shape[2], phi_half_size, phi_half_size)
        return phi  # new: r_n, mc, out_channel, r_h, r_w


# W
class Conv_Linear(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=0, mc=10, group=1,
                 local_reparam=True, F0=False):
        super(Conv_Linear, self).__init__()

        self.in_channels = in_channels // group
        self.out_channels = out_channels
        self.kernel_size = k_size
        self.stride = stride
        self.padding = padding
        self.mc = mc
        self.group = group
        self.local_reparam = local_reparam
        self.F0 = F0

        # scaler
        self.W_mean_prior = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.W_mean = nn.Parameter(
            torch.ones(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels), requires_grad=True)
        # scaler
        self.W_logsigma_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        # [C*k*k,C']
        self.W_logsigma = nn.Parameter(
            torch.ones(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels), requires_grad=True)

        if local_reparam:
            self.register_parameter('W_eps', None)
        else:
            # W:[C*k*k,C']
            self.W_eps = nn.Parameter(
                torch.randn(self.mc, self.in_channels * self.kernel_size * self.kernel_size, self.out_channels),
                requires_grad=True)

    def reset_parameters(self, input):
        self.W_eps = nn.Parameter(input.new(input.size()).normal_(0, 1), requires_grad=True)

    def forward(self, phi):
        if self.F0:
            phi = torch.unsqueeze(phi, 0).repeat(self.mc, 1, 1, 1, 1)
            phi = phi.transpose(0, 1)  # new: r_n, mc, out_channel, r_h, r_w

        # local reparam
        if self.local_reparam:
            # [mc,C*k*k,C']
            # W_mean = torch.unsqueeze(self.W_mean, 0).repeat(self.mc, 1, 1)
            # W_logsigma = torch.unsqueeze(self.W_logsigma, 0).repeat(self.mc, 1, 1)
            W_mean = self.W_mean.permute(1, 0)
            W_logsigma = self.W_logsigma.permute(1, 0)

            phi = _5d_to_4d(phi, use_reshape=True)
            W_mean_F = F.conv2d(phi, W_mean, stride=self.stride, padding=self.padding)
            W_logsigma_F = F.conv2d(torch.pow(phi, 2), torch.exp(W_logsigma), stride=self.stride, padding=self.padding)

            if self.W_eps is None:
                self.reset_parameters(W_logsigma_F)
            F_y = W_mean_F + torch.sqrt(W_logsigma_F) * self.W_eps
            F_y = _4d_to_5d(F_y, self.mc)
        else:
            W_from_q = (self.W_mean + torch.exp(self.W_logsigma / 2.) * self.W_eps)
            ## conv2d
            # phi = self.quant(phi)
            # r_n, mc, out_channel, r_h, r_w
            F_y = conv2d(phi, W_from_q, self.in_channels, self.out_channels, self.kernel_size, self.stride,
                         self.padding, self.group)
            # F_y = self.dequant(F_y)
        return F_y


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, mc=10, kernel_type='RBF', group=1, head=False):
        super().__init__()
        padding = k_size // 2

        self.stride = stride
        self.rff = Conv_RFF(in_channels, out_channels, k_size, stride, padding, mc, kernel_type, group, head)
        self.linear = Conv_Linear(out_channels, out_channels, 3, 1, 1, mc, group, False)

    def forward(self, x):
        return self.linear(self.rff(x))


# W
class FullyConnected(nn.Module):
    def __init__(self, in_features, out, mc=10, local_reparam=False):
        super(FullyConnected, self).__init__()
        self.in_features = in_features
        self.out = out
        self.mc = mc
        self.local_reparam = local_reparam
        # [C*H*W,C']
        self.W_mean_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.W_mean = nn.Parameter(torch.ones(self.in_features, self.out), requires_grad=True)
        self.W_logsigma_prior = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.W_logsigma = nn.Parameter(torch.ones(self.in_features, self.out), requires_grad=True)

        if local_reparam:
            # [mc,batch_size,C']
            # self.W_eps = nn.Parameter(torch.randn(self.mc, -1, self.out), requires_grad=True)
            self.W_eps = nn.Parameter(torch.randn(self.mc, 1, self.out), requires_grad=True)
        else:
            # [mc,C*H*W,C']
            self.W_eps = nn.Parameter(torch.randn(self.mc, self.in_features, self.out), requires_grad=True)

    def forward(self, phi):
        # print("phi",phi)
        # local reparam
        if self.local_reparam:

            # W_eps = nn.Parameter(torch.randn(self.mc, 1, self.out)).cuda()
            # [mc,C*H*W,C']
            W_mean = torch.unsqueeze(self.W_mean, 0).repeat(self.mc, 1, 1)
            W_logsigma = torch.unsqueeze(self.W_logsigma, 0).repeat(self.mc, 1, 1)
            # phi:[mc,batch_size,C*H*W]
            # W_mean:[mc,C*H*W,C']
            W_mean_F = torch.matmul(phi, W_mean)
            W_logsigma_F = torch.matmul(torch.pow(phi, 2), torch.exp(W_logsigma))
            # F_y:[mc,batch_size,C']
            F_y = W_mean_F + torch.sqrt(W_logsigma_F) * self.W_eps
        else:
            # [mc,C*H*W,C']
            W_from_q = (self.W_mean + torch.exp(self.W_logsigma / 2.) * self.W_eps)
            # phi:[mc,batch_size,C*H*W]
            # W_from_q:[mc,C*H*W,C']
            # F_y:[mc,batch_size,C']
            F_y = torch.matmul(phi, W_from_q)
        return F_y


class Resnet(nn.Module):
    """Implement for Resnet"""

    def __init__(self, in_chanel=3, k_size=3, stride=1):
        super(Resnet, self).__init__()
        self.in_ch = in_chanel
        padding = k_size // 2

        # self.cbam = CBAM(in_chanel, in_chanel)

        self.pool = nn.AvgPool2d(1) if stride == 1 else nn.AvgPool2d(k_size, stride, padding)
        # self.pool = nn.functional.interpolate(downsample)

    def forward(self, x, F):
        mc = x.size(1)
        x = _5d_to_4d(x)
        # print(x.shape)
        # if self.in_ch !=3:
        #     x = self.cbam(x)
        x = self.pool(x)

        x = _4d_to_5d(x, mc)
        # x = nn.functional.interpolate(x,self.downsample)
        F_y = torch.cat([x, F], 2)
        # F_y = torch.transpose(F_y, 1, 2).contiguous()
        # F_y = torch.add(x, F)
        return F_y


def conv2d(x, weight, in_channel, out_channel, k_size=3, stride=2, padding=0, group=1):

    mc = x.size(1)

    x = _5d_to_4d(x, use_reshape=True)
    weight = weight.view(mc, k_size, k_size, in_channel, out_channel)
    weight = weight.permute(0, 4, 3, 1, 2)
    weight = weight.reshape(out_channel * mc, in_channel, k_size, k_size)

    result = F.conv2d(x, weight, stride=stride, groups=mc * group, padding=padding)

    r_n, _, r_h, r_w = result.shape
    result = result.view(r_n, mc, out_channel, r_h, r_w)
    return result


def _4d_to_5d(x, mc, use_reshape=False):
    n, c, h, w = x.shape
    assert c % mc == 0
    if use_reshape:
        x = x.reshape(n, mc, c // mc, h, w)
    else:
        x = x.view(n, mc, c // mc, h, w)
    return x


def _5d_to_4d(x, use_reshape=False):
    n, mc, c, h, w = x.shape
    if use_reshape:
        x = x.reshape(n, mc * c, h, w)
    else:
        x = x.view(n, mc * c, h, w)
    return x
