import torch
import commons
import utils
import os
import sys
from models import EFTS2

from scipy.io.wavfile import write
from text.symbols import symbols
from text import text_to_sequence, cleaned_text_to_sequence

class Synthesizer:
  def __init__(self, path):
    if path.endswith('.pth'):
      self.model_dir = os.path.dirname(path)
      self.ckpt_path = path
    else:
      self.model_dir = path
      self.ckpt_path = self.get_model()
    self.load_model()

  def get_model(self):
    return max(os.listdir(self.model_dir), key=lambda x: int(x[2:-4]))

  def load_model(self):
    print('loading model from {}...'.format(self.ckpt_path))
    self.hps = utils.get_hparams_from_dir(self.model_dir)
    print(self.hps)
    self.net_g = EFTS2(
      len(symbols), self.hps.data.filter_length // 2 + 1, self.hps.train.segment_size // self.hps.data.hop_length,
      n_speakers=self.hps.data.n_speakers, **self.hps.model
    ).cuda()
    self.net_g.eval()

  def text_to_tensor(self, text):
    text_norm = cleaned_text_to_sequence(text)
    if self.hps.data.add_blank:
      text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

  def synthesize(self, text_path, out_dir='output'):
    os.makedirs(out_dir, exist_ok=True)
    text_list = utils.load_filepaths_and_text(text_path)
    for id, text in text_list:
      real_id = id.split('/')[-1].replace('.wav', '')
      text_stn = self.text_to_tensor(text)
      with torch.no_grad():
        x_tst = text_stn.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([text_stn.size(0)]).cuda()
        audio = self.net_g.infer(x_tst, x_tst_lengths, t1=.667, t2=0.7, length_scale=1, ta=0.7, max_len=2000)[0][0,0].data.cpu().float().numpy()
        write('{}/{}.wav'.format(out_dir, real_id), self.hps.data.sampling_rate, audio)


if __name__ == '__main__':
  syn = Synthesizer(sys.argv[1])
  syn.synthesize(sys.argv[2])

