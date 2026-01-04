import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class LF_ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(1, 0.25), res_scale=1):
        super(LF_ResBlock, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv4 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)

    def forward(self, x):
        yn = x
        G_yn = self.relu1(x)
        G_yn = self.conv1(G_yn)
        yn_1 = G_yn * self.scale1
        Gyn_1 = self.relu2(yn_1)
        Gyn_1 = self.conv2(Gyn_1)
        yn_2 = Gyn_1 * self.scale2
        yn_2 = yn_2 + yn
        Gyn_2 = self.relu3(yn_2)
        Gyn_2 = self.conv3(Gyn_2)
        yn_3 = Gyn_2 * self.scale3
        yn_3 = yn_3 + yn_1
        Gyn_3 = self.relu4(yn_3)
        Gyn_3 = self.conv4(Gyn_3)
        yn_4 = Gyn_3 * self.scale4
        out = yn_4 + yn_2
        return out



class RK_ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(1, 0.25), res_scale=1):
        super(RK_ResBlock, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)
        self.scale5 = nn.Parameter(torch.FloatTensor([1 / 6]), requires_grad=True)

    def forward(self, x):
        yn = x
        k1 = self.relu1(x)
        k1 = self.conv1(k1)
        yn_1 = k1 * self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        yn_2 = yn + self.scale2 * k2
        yn_2 = yn_2 + k1 * self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        yn_3 = k3 + k2 * self.scale4 + k1
        yn_3 = yn_3 * self.scale5
        out = yn_3 + yn
        return out



class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

from IPython import embed
class ResBlock_shift(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, move_pixel=0, move_channel=0):

        super(ResBlock_shift, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.m_p = move_pixel # 2
        self.m_c = move_channel # 8

    def shift_uni_features(self, input, move_pixel, move_channel=0, direction='H+'):

        if move_channel % 2 == 0:
            H = input.shape[2]
            W = input.shape[3]
            channel_size = input.shape[1]
            mid_channel = channel_size // 2

            zeros = torch.zeros_like(input[:, :move_channel])
            if direction == 'H+':  #this lwj
                zeros[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                               move_pixel:, :]  # up
            elif direction == 'H-':
                zeros[:, :, move_pixel:, :] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                              :H - move_pixel, :]  # down
            elif direction == 'W+':
                zeros[:, :, :, move_pixel:] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                              :,
                                              :W - move_pixel]  # right
            elif direction == 'W-':
                zeros[:, :, :, :-move_pixel] = input[:, mid_channel - move_channel // 2:mid_channel + move_channel // 2,
                                               :,
                                               move_pixel:]  # left
            else:
                raise NotImplementedError("Direction should be 'H+', 'H-', 'W+', 'W-'.")

            return torch.cat(
                (input[:, 0:mid_channel - move_channel // 2], zeros, input[:, mid_channel + move_channel // 2:]),
                1)

        elif move_channel % 2 != 0:
            H = input.shape[2]
            W = input.shape[3]
            channel_size = input.shape[1]
            mid_channel = channel_size // 2

            zeros = torch.zeros_like(input[:, :move_channel])
            if direction == 'H+':  #this lwj
                zeros[:, :, :-move_pixel, :] = input[:,
                                               mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                               move_pixel:, :]  # up
            elif direction == 'H-':
                zeros[:, :, move_pixel:, :] = input[:,
                                              mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                              :H - move_pixel, :]  # down
            elif direction == 'W+':
                zeros[:, :, :, move_pixel:] = input[:,
                                              mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                              :,
                                              :W - move_pixel]  # right
            elif direction == 'W-':
                zeros[:, :, :, :-move_pixel] = input[:,
                                               mid_channel - move_channel // 2:mid_channel + move_channel // 2 + 1,
                                               :,
                                               move_pixel:]  # left
            else:
                raise NotImplementedError("Direction should be 'H+', 'H-', 'W+', 'W-'.")

            return torch.cat(
                (input[:, 0:mid_channel - move_channel // 2], zeros, input[:, mid_channel + move_channel // 2 + 1:]),
                1)
            
    def forward(self, x):
        x1 = self.shift_uni_features(x, self.m_p, self.m_c, 'H+')
        # embed()
        res = self.body(x1).mul(self.res_scale)
        res += x

        return res


class ResBlock_shift_bi(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, move_pixel=0, move_channel=0):

        super(ResBlock_shift_bi, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.m_p = move_pixel
        self.m_c = move_channel

    def shift_bi_features(self, input, move_pixel, move_channel=0, direction='H'):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left = torch.zeros_like(input[:, :move_channel])
        zero_right = torch.zeros_like(input[:, :move_channel])
        if direction == 'H':
            zero_left[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel:mid_channel, move_pixel:, :]  # up
            zero_right[:, :, move_pixel:, :] = input[:, mid_channel:mid_channel + move_channel, :H - move_pixel,
                                               :]  # down

        elif direction == 'W':
            zero_left[:, :, :, :-move_pixel] = input[:, mid_channel - move_channel:mid_channel, :, move_pixel:]  # left
            zero_right[:, :, :, move_pixel:] = input[:, mid_channel:mid_channel + move_channel, :,
                                               :W - move_pixel]  # right

        else:
            raise NotImplementedError("Direction should be 'H' or 'W'.")
        return torch.cat(
            (input[:, 0:mid_channel - move_channel], zero_left, zero_right, input[:, mid_channel + move_channel:]), 1)

    def forward(self, x):
        x1 = self.shift_bi_features(x, self.m_p, self.m_c, 'H')
        res = self.body(x1).mul(self.res_scale)
        res += x
        return res


class ResBlock_shift_cross(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, move_pixel=0, move_channel=0):

        super(ResBlock_shift_cross, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.m_p = move_pixel
        self.m_c = move_channel
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def shift_cross_features(self, input, move_pixel=0, move_channel=0, w='+', h='+'):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left = torch.zeros_like(input[:, :move_channel])
        zero_right = torch.zeros_like(input[:, :move_channel])
        if h == '+':
            zero_left[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel:mid_channel, move_pixel:, :]  # up
        elif h == '-':
            zero_left[:, :, move_pixel:, :] = input[:, mid_channel - move_channel:mid_channel, :H - move_pixel,
                                              :]  # down
        else:
            raise NotImplementedError("Direction on H should be '+' or '-'.")
        if w == '+':
            zero_right[:, :, :, move_pixel:] = input[:, mid_channel:mid_channel + move_channel, :,
                                               :W - move_pixel]  # right
        elif w == '-':
            zero_right[:, :, :, :-move_pixel] = input[:, mid_channel:mid_channel + move_channel, :, move_pixel:]  # left
        else:
            raise NotImplementedError("Direction on W should be '+' or '-'.")

        return torch.cat(
            (input[:, 0:mid_channel - move_channel], zero_left, zero_right, input[:, mid_channel + move_channel:]), 1)

    def forward(self, x):
        x1 = self.shift_cross_features(x, self.m_p, self.m_c, w='-', h='+')
        res = self.body(x1).mul(self.res_scale)
        res += x

        return res


class ResBlock_shift_quad(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, move_pixel=0, move_channel=0):

        super(ResBlock_shift_quad, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.m_p = move_pixel
        self.m_c = move_channel
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def shift_all_features(self, input, move_pixel, move_channel=0):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2
        zero_left1 = torch.zeros_like(input[:, :move_channel])
        zero_right1 = torch.zeros_like(input[:, :move_channel])
        zero_left2 = torch.zeros_like(input[:, :move_channel])
        zero_right2 = torch.zeros_like(input[:, :move_channel])

        zero_left3 = torch.zeros_like(input[:, :move_channel])
        zero_right3 = torch.zeros_like(input[:, :move_channel])
        zero_left4 = torch.zeros_like(input[:, :move_channel])
        zero_right4 = torch.zeros_like(input[:, :move_channel])

        zero_left1[:, :, :-move_pixel, :] = input[:, mid_channel - move_channel * 2:mid_channel - move_channel,
                                            move_pixel:, :]  # up
        zero_left2[:, :, move_pixel:, :] = input[:, mid_channel - move_channel:mid_channel, :H - move_pixel, :]  # down
        zero_left3[:, :, :-move_pixel, move_pixel:] = input[:,
                                                      mid_channel - move_channel * 3:mid_channel - move_channel * 2,
                                                      move_pixel:, :W - move_pixel]  # 1
        zero_left4[:, :, :-move_pixel, :-move_pixel] = input[:,
                                                       mid_channel - move_channel * 4:mid_channel - move_channel * 3,
                                                       move_pixel:, move_pixel:]  # 2
        zero_right1[:, :, :, :-move_pixel] = input[:, mid_channel:mid_channel + move_channel, :, move_pixel:]  # left
        zero_right2[:, :, :, move_pixel:] = input[:, mid_channel + move_channel:mid_channel + move_channel * 2, :,
                                            :W - move_pixel]  # right
        zero_right3[:, :, move_pixel:, :-move_pixel] = input[:,
                                                       mid_channel + move_channel * 2:mid_channel + move_channel * 3,
                                                       :H - move_pixel, move_pixel:]  # left
        zero_right4[:, :, move_pixel:, move_pixel:] = input[:,
                                                      mid_channel + move_channel * 3:mid_channel + move_channel * 4,
                                                      :H - move_pixel, :W - move_pixel]  # right

        return torch.cat(
            (input[:, 0:mid_channel - move_channel * 4], zero_left4, zero_left3, zero_left1, zero_left2, zero_right1,
             zero_right2, zero_right3, zero_right4,
             input[:, mid_channel + move_channel * 4:]),
            1)

    def shift_quad_features(self, input, move, m_c=0):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left1 = torch.zeros_like(input[:, :m_c])
        zero_right1 = torch.zeros_like(input[:, :m_c])
        zero_left2 = torch.zeros_like(input[:, :m_c])
        zero_right2 = torch.zeros_like(input[:, :m_c])

        zero_left1[:, :, :-move, :] = input[:, mid_channel - m_c * 2:mid_channel - m_c, move:, :]  # up
        zero_left2[:, :, move:, :] = input[:, mid_channel - m_c:mid_channel, :H - move, :]  # down
        zero_right1[:, :, :, :-move] = input[:, mid_channel:mid_channel + m_c, :, move:]  # left
        zero_right2[:, :, :, move:] = input[:, mid_channel + m_c:mid_channel + m_c * 2, :, :W - move]  # right

        return torch.cat(
            (input[:, 0:mid_channel - m_c * 2], zero_left1, zero_left2, zero_right1, zero_right2,
             input[:, mid_channel + m_c * 2:]),
            1)

    def forward(self, x):
        # x1 = self.shift_all_features(x, self.m_p, self.m_c)
        x1 = self.shift_quad_features(x, self.m_p, self.m_c)
        res = self.body(x1).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


class ResBlock_large_kernel(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_large_kernel, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.large_kernel = conv(n_feats, n_feats, kernel_size=11, bias=bias)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = self.large_kernel(res)
        res += x

        return res


'''
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim,dim*2,1,1,0)
        self.linear_1 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_2 = nn.Conv2d(dim,dim,1,1,0)

        self.lde = DMlp(dim,2)

        self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))

    def forward(self, f):
        _,_,h,w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h,w), mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)
'''

############################################################################
from inspect import isfunction
def exists(val):
    return val is not None

def is_empty(t):
    return t.nelement() == 0

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))
    
    
def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def center_iter(x, means, buckets = None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means
    
from einops import rearrange
class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size
        
    
    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape
       
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))
   
        gs = min(N, self.group_size)  # group size
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N
        
        paded_q = torch.cat((q, torch.flip(q[:,N-pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d",ng=ng,h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2,2*gs,gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2,2*gs,gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        out1 = F.scaled_dot_product_attention(paded_q,paded_k,paded_v)
        
        
        k_global = k_global.reshape(1,1,*k_global.shape).expand(B,ng,-1,-1,-1)
        v_global = v_global.reshape(1,1,*v_global.shape).expand(B,ng,-1,-1,-1)
       
        out2 = F.scaled_dot_product_attention(paded_q,k_global,v_global)
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
 
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)
    
        return out
    
class IRCA(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
      
    def forward(self, normed_x, x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))
        else:
            x_global = x_means

        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k,v, x_global.detach()
    
class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x
    
class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x
    
class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class TAB(nn.Module):
    def __init__(self, dim, qk_dim=60, mlp_dim=180, heads=4, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay = 0.999):
        super().__init__()

        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens
        
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim,mlp_dim))
        self.irca_attn = IRCA(dim,qk_dim,heads)
        self.iasa_attn = IASA(dim,qk_dim,heads,group_size)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim,dim,1, bias=False)

    
    def forward(self, x):
        _,_,h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape
        
        idx_last = torch.arange(N, device=x.device).reshape(1,N).expand(B,-1)
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:,N-pad_n:N, :], dims=[-2])), dim=-2)
            x_means=torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c',cnt=self.num_tokens),dim=-2).detach()   
        else:  
            x_means = self.means.detach()

        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter-1):
                    x_means = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))
                        
                
        k_global, v_global, x_means = self.irca_attn(x, x_means)
        
        with torch.no_grad():
            x_scores = torch.einsum('b i c,j c->b i j', 
                                        F.normalize(x, dim=-1), 
                                        F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)
    
            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)
        
        y = self.iasa_attn(x, idx_last,k_global,v_global)
        y = rearrange(y,'b (h w) c->b c h w',h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x
        
 
        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else: 
                    ema_inplace(self.means, new_means, self.ema_decay)
            
    
        return rearrange(x, 'b (h w) c->b c h w',h=h)
############################################################################
    
############################################################################
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio+3,
                           stride=sr_ratio,
                           padding=(sr_ratio+3)//2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None,),)
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C//self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:], 
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)

class DynamicConv2d(nn.Module): ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=2,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim, 
                       dim//reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'),),
            nn.Conv2d(dim//reduction_ratio, dim*num_groups, kernel_size=1),)

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):

        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K//2,
                     groups=B*C,
                     bias=bias)
        
        return x.reshape(B, C, H, W)

class HybridTokenMixer(nn.Module): ### D-Mixer
    def __init__(self, 
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim//2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(
            dim=dim//2, num_heads=num_heads, sr_ratio=sr_ratio)
        
        inner_dim = max(16, dim//reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),)

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x ## STE
        return x
#############################################################################

from features import visualize_feature_map
#############################################################################
class ResBlock_fre(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_fre, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.fre = Frequency_Convolution(n_feats)


    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = self.fre(res)
        res += x

        return res


class Frequency_Convolution(nn.Module):
    """
    channels: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """
    def __init__(self, channels, num_blocks=8, sparsity_threshold=0.01):
        super().__init__()
        assert channels % num_blocks == 0, f"channels {channels} should be divisble by num_blocks {num_blocks}"

        self.channels = channels
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = channels // self.num_blocks
        self.scale = 0.02

        self.w = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size, 2))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.b = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        weight = torch.view_as_complex(self.w.contiguous())
        x = torch.einsum('bkihw,kio->bkohw', x, weight)

        o1_real = F.relu(
            torch.mul(x.real, self.w1[0].unsqueeze(dim=0)) - \
            torch.mul(x.imag, self.w1[1].unsqueeze(dim=0)) + \
            self.b[0, :, :, None, None]
        ) # [16, 8, 8, 48, 25]  x.imag=[16, 8, 8, 48, 25]
        
        o1_imag = F.relu(
            torch.mul(x.imag, self.w2[0].unsqueeze(dim=0)) + \
            torch.mul(x.real, self.w2[1].unsqueeze(dim=0)) + \
            self.b[1, :, :, None, None]
        ) # [16, 8, 8, 48, 25] x.real=[16, 8, 8, 48, 25]

        x = torch.stack([o1_real, o1_imag], dim=-1) # [16, 8, 8, 48, 25, 2]
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x) # [16, 8, 8, 48, 25]
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias




'''
class AFNO2D_channelfirst(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.fliter = nn.Parameter(self.scale * torch.randn(1, self.hidden_size, 1, 1))

    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        o1_real = F.relu(
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1]) + \
            self.b1[0, :, :, None, None]
        ) # [16, 8, 8, 48, 25]  x.imag=[16, 8, 8, 48, 25]

        o1_imag = F.relu(
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1]) + \
            self.b1[1, :, :, None, None]
        ) # [16, 8, 8, 48, 25]


        x = torch.stack([o1_real, o1_imag], dim=-1) # [16, 8, 8, 48, 25, 2]
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x) # [16, 8, 8, 48, 25]

        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = x * self.fliter + origin_ffted 

        # x = x + origin_ffted 

        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias
'''

'''
class AFNO2D_channelfirst(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.fliter = nn.Parameter(self.scale * torch.randn(1, hidden_size, 1, 1))

    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        o1_real = F.relu(
            torch.mul(x.real, self.w1[0].unsqueeze(dim=0)) - \
            torch.mul(x.imag, self.w1[1].unsqueeze(dim=0)) + \
            self.b1[0, :, :, None, None]
        ) # [16, 8, 8, 48, 25]  x.imag=[16, 8, 8, 48, 25]
        # print(x.real.size())
        # print(self.w1[0].size())
        # print(self.b1[0, :, :, None, None].size())
        # print(o1_real.size())

        o1_imag = F.relu(
            torch.mul(x.imag, self.w2[0].unsqueeze(dim=0)) + \
            torch.mul(x.real, self.w2[1].unsqueeze(dim=0)) + \
            self.b1[1, :, :, None, None]
        ) # [16, 8, 8, 48, 25]


        x = torch.stack([o1_real, o1_imag], dim=-1) # [16, 8, 8, 48, 25, 2]
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x) # [16, 8, 8, 48, 25]
        x = x.reshape(B, C, x.shape[3], x.shape[4])


        x = x * self.fliter + origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias
'''

