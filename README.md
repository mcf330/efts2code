# EfficientTTS 2: Variational End-to-End Text-to-Speech Synthesis and Voice Conversion

This is a implementation of our work: EfficientTTS 2: Variational End-to-End Text-to-Speech Synthesis and Voice Conversion

This implementation uses code from VITS repos: https://github.com/jaywalnut310/vits

## Pre-requisites
0. Python >= 3.6
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For voice conversion setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
    1. Extract the speaker embedings of each audio following the instructions from YourTTS https://github.com/freds0/YourTTS
```sh

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Exmaple
```sh
# TTS on LJ-Speech (EFTS2)
python train.py -c configs/ljs_base.json -m ljs_base

# VC on VCTK (EFTS2-VC)
python train_ms_vc.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
# TTS LJ-Speech
python3 inference.py path-to-checkpoint filelists/ljs_audio_text_test_filelist.txt.cleaned
# VC VCTK
python3 inference_vc.py path-to-checkpoint filelists/vctk_audio_sid_text_test_filelist.txt.cleaned 

