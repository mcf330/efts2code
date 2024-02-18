import copy
import math
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm

import commons
from commons import get_padding, init_weights
import attentions


LRELU_SLOPE = 0.1
 
class WN(torch.nn.Module):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    output = torch.zeros_like(x)
    n_channels_tensor = torch.IntTensor([self.hidden_channels])

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        g_l = torch.zeros_like(x_in)

      acts = commons.fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          n_channels_tensor)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        output = output + res_skip_acts
    return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HybridAttention(nn.Module):
    def __init__(self, channels):
      super().__init__()
      self.sigma = nn.Parameter(torch.ones(2)*0.1)
      self.linear_hidden = nn.Conv1d(channels,channels*2,1)
      self.linear_out = nn.Conv1d(channels*2,channels,1)

    def forward(self, e, a, b, x_h, text_mask=None, mel_mask=None, max_length=1000, min_length=10):
      if mel_mask is None : # for inference only
          length = torch.round(b[:,-1]).squeeze().item() + 1
          length = min_length if length < min_length else length
          length = max_length if length > max_length else length
      else:
          length = mel_mask.size(-1)
      q = torch.arange(0, length).unsqueeze(0).repeat(e.size(0),1).type_as(e)
      if mel_mask is not None:
          q = q*mel_mask.float()
      energies_e = -1 * (q.unsqueeze(-1) - e.unsqueeze(1))**2
      energies_boundary = -1 * (
           torch.abs(q.unsqueeze(-1) - a.unsqueeze(1)) +
           torch.abs(q.unsqueeze(-1) - b.unsqueeze(1)) -
           (b.unsqueeze(1) - a.unsqueeze(1)) )**2
      energies_all = torch.cat([energies_e.unsqueeze(1), energies_boundary.unsqueeze(1)], 1)
      real_sigma = torch.clamp(self.sigma, max=3, min=1e-6)
      energies = energies_all * real_sigma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
      if text_mask is not None:
          energies = energies.masked_fill(~(text_mask.unsqueeze(1).unsqueeze(1)), -float('inf'))
      attns = torch.softmax(energies, dim=-1)
      h = self.linear_hidden(x_h)
      h = h.view(h.size(0), 2, h.size(1)//2, -1).transpose(2,3)
      out = torch.matmul(attns, h)
      out = out.transpose(2,3).contiguous().view(h.size(0), h.size(3)*2, -1)
      out = self.linear_out(out)
      return out, attns, real_sigma

class AttentionPI(nn.Module):
    def __init__(self, channels, attntion_heads):
      super().__init__()
      self.sigma = nn.Parameter(torch.ones(attntion_heads)*0.1)
      self.attntion_heads = attntion_heads
      self.linear_hidden = nn.Conv1d(channels,channels*2,1)
      self.linear_out = nn.Conv1d(channels*2,channels,1)

    def forward(self, pi, p,  x_h, text_mask=None, mel_mask=None):
      energies = -1 * (pi.unsqueeze(-1) - p.unsqueeze(1))**2
      real_sigma = torch.clamp(self.sigma, max=3, min=1e-6)
      energies = energies.unsqueeze(1) * real_sigma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
      if text_mask is not None:
          energies = energies.masked_fill(~(text_mask.unsqueeze(1).unsqueeze(1)), -float('inf'))
      attns = torch.softmax(energies, dim=-1)
      h = self.linear_hidden(x_h)
      h = h.view(h.size(0), self.attntion_heads, h.size(1)//self.attntion_heads, -1).transpose(2,3)
      out = torch.matmul(attns, h)
      out = out.transpose(2,3).contiguous().view(h.size(0), h.size(3)*self.attntion_heads, -1)
      out = self.linear_out(out)
      return out, attns, real_sigma


class VarationalAlignmentPredictor(nn.Module):
     def __init__(self, filter_channels, kernel_size, n_layers, duration_offset=1.0, gin_channels=0):
         super().__init__()
         self.conv_y = nn.Conv1d(2, filter_channels, 1)
         if gin_channels > 0:
             self.conv_g = nn.Conv1d(gin_channels, filter_channels, 1)
         self.encoder = WN(filter_channels, kernel_size, 1, n_layers, gin_channels=filter_channels)
         self.proj_z = nn.Conv1d(filter_channels, 2*filter_channels, 1)

         self.decoder = WN(filter_channels, kernel_size, 1, n_layers, gin_channels=filter_channels, p_dropout=0.5)
         self.out = nn.Conv1d(filter_channels, 2, 1)

         self.duration_offset = duration_offset

     def forward(self, x, x_mask, e_a, b_a, g=None):
         y_input = torch.detach(torch.cat([e_a.unsqueeze(1), b_a.unsqueeze(1)], 1)) * x_mask
         y = self.conv_y(y_input)

         if g is not None:
             x = x + self.conv_g(g)

         x = torch.detach(x)*x_mask
         y_enc = self.encoder(y, x_mask, g=x)
         stats = self.proj_z(y_enc)
         m, logs = torch.split(stats, stats.size(1)//2, dim=1)
         z = torch.distributions.Normal(m, torch.exp(logs*0.5)).rsample()*x_mask
         dec_y = self.decoder(z,x_mask, g=x)
         y_hat = self.out(dec_y)

         loss_dur = torch.sum((torch.log(y_input + self.duration_offset)*x_mask - y_hat*x_mask)**2) / torch.sum(x_mask)
         loss_kl = torch.sum( torch.distributions.kl_divergence(torch.distributions.Normal(m, torch.exp(logs*0.5)),
                       torch.distributions.Normal(torch.zeros_like(m), torch.ones_like(m)))* x_mask) / torch.sum(x_mask)
         return loss_dur, loss_kl

     def infer(self, x, x_mask, t_a=1.0, g=None):
         if g is not None:
             x = x + self.conv_g(g)
         x = x * x_mask
         z = torch.randn_like(x)* t_a
         dec_y = self.decoder(z, x_mask, g=x)
         y_hat = self.out(dec_y)
         return torch.clamp(torch.exp(y_hat) - self.duration_offset, min=0) * x_mask


class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels, 1)

  def forward(self, x, x_lengths):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    x = self.proj(x) * x_mask

    return x, x_mask

class SpectrogramEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj_h = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 4, 1)

  def forward(self, y, y_lengths, g=None, t1=1.0, t2=1.0):
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
    y = self.pre(y) * y_mask
    x = self.enc(y, y_mask, g=g)
    y_h = self.proj_h(x) * y_mask
    z_stats = self.proj(x) * y_mask
    m1, logs1, m2, logs2  = torch.split(z_stats, self.out_channels, dim=1)
    z1 =  torch.distributions.Normal(m1, torch.exp(logs1)*t1).rsample()*y_mask
    z2 =  torch.distributions.Normal(m2, torch.exp(logs2)*t2).rsample()*y_mask
    return y_h, z1, z2, m1, logs1, m2, logs2, y_mask


class PriorNN(nn.Module):
  def __init__(self,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, p_dropout=0.2)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, mask, g=None, t=1.0):
    x = self.enc(x, mask, g=g)
    x = self.proj(x) * mask
    m, logs = torch.split(x, self.out_channels, dim=1)
    z =  torch.distributions.Normal(m, torch.exp(logs)*t).rsample()*mask
    return z, m, logs, mask

class AttentionOperator(nn.Module):
  def __init__(self, hid, n_position=1000):
    super().__init__()
    self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, hid))

  def forward(self, x_h, y_h, x_mask, y_mask, sigma):
    return self.compute_e_and_boundaries(x_h, y_h, x_mask, y_mask, sigma)

  def _get_sinusoid_encoding_table(self, n_position, d_hid):
    def get_position_angle_vec(position):
      return [position / np.power(10000, 2*(hid_j // 2)/ d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

  def compute_PI(self, x_h, y_h, x_mask, y_mask):
    x_h = x_h + self.pos_table[:,:x_h.size(-1)].clone().detach().transpose(1, 2)
    y_h = y_h + self.pos_table[:,:y_h.size(-1)].clone().detach().transpose(1, 2)
    scores = torch.bmm(y_h.transpose(1,2), x_h) / np.sqrt(float(x_h.size(1)))
    scores = scores.masked_fill(~x_mask.unsqueeze(1), -float('inf'))
    alpha = torch.softmax(scores, dim=-1)
    p = torch.arange(0, alpha.size(-1)).type_as(alpha).unsqueeze(0) * x_mask.float()
    pi_dummy = torch.bmm(alpha, p.unsqueeze(-1)).squeeze(-1)
    delta_pi = torch.relu(pi_dummy[:,1:]-pi_dummy[:, :-1])
    delta_pi = torch.cat([torch.zeros(alpha.size(0), 1).type_as(alpha), delta_pi], -1) * y_mask.float()
    pi_f = torch.cumsum(delta_pi, -1)* y_mask.float()
    delta_pi_inverse = torch.flip(delta_pi, [-1])
    pi_b = - torch.flip(torch.cumsum(delta_pi_inverse, -1), [-1])
    pi = pi_f + pi_b
    last_pi, _ = torch.max(pi, dim=-1)
    last_pi = torch.clamp(last_pi, min=1e-8).unsqueeze(1)
    first_pi = pi[:,0:1]
    x_lengths = torch.sum(x_mask.float(), -1)
    max_pi = x_lengths.unsqueeze(-1) -1 
    pi = (pi - first_pi) / (last_pi - first_pi) * max_pi
    return pi, p

  def compute_e_and_boundaries(self, x_h, y_h, x_mask, y_mask, sigma=0.2):
    pi, p = self.compute_PI(x_h, y_h, x_mask, y_mask)
    energies = -1 * (pi.unsqueeze(1) - p.unsqueeze(-1))**2 * sigma
    energies = energies.masked_fill(
      ~(y_mask.unsqueeze(1).repeat(1, energies.size(1), 1)), -float('inf')
    )
    beta = torch.softmax(energies, dim=2)
    q = torch.arange(0, y_mask.size(-1)).unsqueeze(0).type_as(pi) * y_mask.float()
    e = torch.bmm(beta, q.unsqueeze(-1)).squeeze(-1) * x_mask.float()

    boundary_idx = torch.clamp(p-0.5, min=0)
    energies_a = -1 *(pi.unsqueeze(1) - boundary_idx.unsqueeze(-1))**2 * sigma
    energies_a = energies_a.masked_fill(
      ~(y_mask.unsqueeze(1).repeat(1, energies_a.size(1), 1)), -float('inf')
    )
    beta_a = torch.softmax(energies_a, dim=2)
    a = torch.bmm(beta_a, q.unsqueeze(-1)).squeeze(-1) * x_mask.float()
    a_real = torch.cat([torch.zeros(a.size(0), 1).type_as(a), a[:,1:]], -1)

    max_x = torch.sum(x_mask, dim=-1) - 1
    max_y = torch.sum(y_mask, dim=-1) - 1
    b = torch.cat([a_real[:,1:], torch.zeros(a.size(0), 1).type_as(a)], -1)
    b_real = b.scatter_(1, max_x.unsqueeze(1), max_y.unsqueeze(1).type_as(b))

    return e, a_real, b_real






  









