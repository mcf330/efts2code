import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules


class EFTS2(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    spec_encoder_layers,
    prior_nn1_layers,
    prior_nn2_layers,
    vap_layers,
    n_speakers=0,
    gin_channels=0,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels

    self.enc_p = modules.TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    self.dec = modules.Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = modules.SpectrogramEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, spec_encoder_layers, gin_channels=gin_channels)
    self.attn = modules.HybridAttention(inter_channels)
    self.prior_nn1 = modules.PriorNN(inter_channels, hidden_channels, 5, 1, prior_nn1_layers, gin_channels=gin_channels)
    self.prior_nn2 = modules.PriorNN(inter_channels, hidden_channels, 5, 1, prior_nn2_layers, gin_channels=gin_channels)
    self.VAP = modules.VarationalAlignmentPredictor(hidden_channels, 3, n_layers=vap_layers, gin_channels=gin_channels)
    self.attn_op = modules.AttentionOperator(hidden_channels)

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

  def forward(self, x, x_lengths, y, y_lengths, sid=None, bi=False):

    x_h, x_mask = self.enc_p(x, x_lengths)
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    y_h, z1, z2, m1, logs1, m2, logs2, y_mask = self.enc_q(y, y_lengths, g=g)
    x_mask_b, y_mask_b = x_mask.squeeze(1).bool(),  y_mask.squeeze(1).bool()
    
    e, a, b = self.attn_op(x_h, y_h, x_mask_b, y_mask_b, sigma=0.5)
    x_align, attns, real_sigma = self.attn(e, a, b, x_h, x_mask_b, y_mask_b)

    z_slice, ids_slice = commons.rand_slice_segments(z2, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)

    z1_r, m_q1, logs_q1, y_mask= self.prior_nn1(x_align, y_mask, g=g)
    _, m_q2, logs_q2, y_mask= self.prior_nn2(z1, y_mask, g=g)
    
    e_a = e - a
    b_a = b - a
    loss_a, loss_a_kl = self.VAP(x_h, x_mask, e_a, b_a, g=g)
    if bi is True:
      _, m_q2_r, logs_q2_r, y_mask= self.prior_nn2(z1_r, y_mask, g=g)
      return o, (loss_a, loss_a_kl), attns, ids_slice, x_mask, y_mask, (m1, logs1, m_q1, logs_q1), (m2, logs2, m_q2, logs_q2), (m2, logs2, m_q2_r, logs_q2_r), real_sigma
    else:
      return o, (loss_a, loss_a_kl), attns, ids_slice, x_mask, y_mask, (m1, logs1, m_q1, logs_q1), (m2, logs2, m_q2, logs_q2), None, real_sigma

  def infer(self, x, x_lengths, sid=None, t1=0.7, t2=0.7, length_scale=1.0, ta=0.7, max_len=2000):
    x_h, x_mask = self.enc_p(x, x_lengths)
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None
    vap_outs = self.VAP.infer(x_h, x_mask, t_a=0.7, g=g)
    b = torch.cumsum(vap_outs[:,1,:], dim=1) * length_scale
    a = torch.cat([torch.zeros(b.size(0), 1).type_as(b), b[:,:-1]], -1)
    e = a + vap_outs[:,0,:] * length_scale
    x_align, attns, real_sigma = self.attn(e, a, b, x_h, text_mask=None, mel_mask=None, max_length = max_len)
    y_mask = torch.ones(x_align.size(0), 1, x_align.size(-1)).type_as(x_mask)
    z_1, _, _, _= self.prior_nn1(x_align, y_mask, g=g, t=t1)
    z_2, _, _, _= self.prior_nn2(z_1, y_mask, g=g, t=t2)
    o = self.dec(z_2, g=g)
    return o, attns, y_mask


class EFTS2VC(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    spec_encoder_layers,
    prior_nn1_layers,
    prior_nn2_layers,
    n_speakers=0,
    gin_channels=0,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels

    self.enc_p = modules.TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    self.dec = modules.Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = modules.SpectrogramEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, spec_encoder_layers, gin_channels=gin_channels)
    self.attn = modules.AttentionPI(inter_channels, 4)
    self.prior_nn1 = modules.PriorNN(inter_channels, hidden_channels, 5, 1, prior_nn1_layers, gin_channels=gin_channels)
    self.prior_nn2 = modules.PriorNN(inter_channels, hidden_channels, 5, 1, prior_nn2_layers, gin_channels=gin_channels)
    self.attn_op = modules.AttentionOperator(hidden_channels)

  def forward(self, x, x_lengths, y, y_lengths, sp_embed):

    x_h, x_mask = self.enc_p(x, x_lengths)
    g = sp_embed.unsqueeze(-1)

    y_h, z1, z2, m1, logs1, m2, logs2, y_mask = self.enc_q(y, y_lengths, g=g)
    x_mask_b, y_mask_b = x_mask.squeeze(1).bool(),  y_mask.squeeze(1).bool()
    
    pi, p = self.attn_op.compute_PI(x_h, y_h, x_mask_b, y_mask_b)
    x_align, attns, real_sigma = self.attn(pi, p, x_h, x_mask_b, y_mask_b)

    z_slice, ids_slice = commons.rand_slice_segments(z2, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)

    _, m_q1, logs_q1, y_mask= self.prior_nn1(x_align, y_mask, g=None)
    _, m_q2, logs_q2, y_mask= self.prior_nn2(z1, y_mask, g=g)

    return o, attns, ids_slice, x_mask, y_mask, (m1, logs1, m_q1, logs_q1), (m2, logs2, m_q2, logs_q2), real_sigma

  def infer(self, y, y_lengths, y_emb, tgt_emb, t1=0.7, t2=0.7):
    y_emb, tgt_emb = y_emb.unsqueeze(-1), tgt_emb.unsqueeze(-1)
    y_h, z1, z2, m1, logs1, m2, logs2, y_mask = self.enc_q(y, y_lengths, g=y_emb, t1=t1)
    z2, m_q2, logs_q2, y_mask= self.prior_nn2(z1, y_mask, g=tgt_emb, t=t2)
    o = self.dec(z2, g=tgt_emb)
    return o






