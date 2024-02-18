import torch
import commons
import utils
import os
import sys
from models import EFTS2VC

from scipy.io.wavfile import write
from text.symbols import symbols
from text import text_to_sequence, cleaned_text_to_sequence
from mel_processing import spectrogram_torch

class Synthesizer:
  def __init__(self, path):
    if path.endswith('.pth'):
      self.model_dir = os.path.dirname(path)
      self.ckpt_path = path
    else:
      self.model_dir = path
      self.ckpt_path = self.get_model()
    self.net_g = self.load_model()

  def get_model(self):
    return max(os.listdir(self.model_dir), key=lambda x: int(x[2:-4]))

  def load_model(self):
    print('loading model from {}...'.format(self.ckpt_path))
    self.hps = utils.get_hparams_from_dir(self.model_dir)
    print(self.hps)
    net_g = EFTS2VC(
      len(symbols), self.hps.data.filter_length // 2 + 1, self.hps.train.segment_size // self.hps.data.hop_length,
      n_speakers=self.hps.data.n_speakers, **self.hps.model
    ).cuda()
    net_g.eval()
    utils.load_checkpoint(self.ckpt_path, net_g, None)
    return net_g

  def text_to_tensor(self, text):
    text_norm = cleaned_text_to_sequence(text)
    if self.hps.data.add_blank:
      text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

  def get_audio(self, filename):
    audio, _ = utils.load_wav_to_torch(filename)
    audio_norm = audio / 32678.0
    audio_norm = audio_norm.unsqueeze(0)
    spec_filename = filename.replace(".wav", ".spec.pt")
    if os.path.exists(spec_filename):
      spec = torch.load(spec_filename)
    else:
      spec = spectrogram_torch(audio_norm, self.hps.data.filter_length,
          self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
            center=False)
      spec = torch.squeeze(spec, 0)
    sp_embed = torch.load(filename.replace('.wav', '_embed.pt')) # hack here, it depends on your speaker embeding dirs
    return spec, audio_norm, sp_embed

  def synthesize(self, text_path, out_dir='output'):
    os.makedirs(out_dir, exist_ok=True)
    text_list = utils.load_filepaths_and_text(text_path)
    for i in range(len(text_list)//2):
      src_spec, _, src_emb = self.get_audio(text_list[2*i][0])
      tgt_spec, _, tgt_emb = self.get_audio(text_list[2*i+1][0])

      real_id_src = text_list[2*i][0].split('/')[-1].replace('.wav', '')
      real_id_tgt = text_list[2*i+1][0].split('/')[-1].replace('.wav', '')

      with torch.no_grad():
        src_spec = src_spec.cuda().unsqueeze(0)
        src_spec_lengths = torch.LongTensor([src_spec.size(-1)]).cuda()
        src_emb = src_emb.cuda().unsqueeze(0)
        tgt_emb = tgt_emb.cuda().unsqueeze(0)

        audio = self.net_g.infer(src_spec, src_spec_lengths, src_emb, tgt_emb, t1=.667, t2=0.7)[0,0].data.cpu().float().numpy()
        write('{}/{}_to_{}.wav'.format(out_dir, real_id_src, real_id_tgt), self.hps.data.sampling_rate, audio)


if __name__ == '__main__':
  syn = Synthesizer(sys.argv[1])
  syn.synthesize(sys.argv[2])

