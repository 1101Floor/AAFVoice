import random
import myconfig
import dataset
import specaug
import numpy as np
import soundfile as sf
import librosa
import decimal
import math
import torch
import logging
from scipy.fft import _pocketfft
import python_speech_features

p_num_colunms = 80
p_lowfreq = 0
p_highfreg = 8000
p_sample_rate = 16000
p_coeff = 0.97
p_winlen = 0.025
p_winstep = 0.01
p_NFFT = 512


# 定义预加重的函数
def preemphasis(signal, coeff=0.95):
    if len(signal.shape) == 2:
        signal = librosa.to_mono(signal.transpose())
    preemphasis_signal = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    return preemphasis_signal


# 定义四舍五入的函数
def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


# 计算分帧加窗的函数
def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))
    padlen = int((numframes - 1) * frame_step + frame_len)
    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len), (numframes, 1))
    return frames * win


# 这个函数用于计算每个帧的幅度谱
def magspec(frames, NFFT):
    if np.shape(frames)[1] > NFFT:
        logging.warn('frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.', np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


# 这个函数用于计算每个帧的功率谱
def powspec(frames, NFFT):
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.)


def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    return _pocketfft.dct(x, type, n, axis, norm, overwrite_x)


def lifter(cepstra, L=22):
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # print(melpoints)
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)
    # print(bin)
    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def create_AAF_matrix(bin, nfft):
    nfilt = len(bin) - 2  # 计算滤波器的数量
    fbank = np.zeros((nfilt, nfft // 2 + 1))
    # 这里使用的是AAF滤波器
    for j in range(nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / ((bin[j + 1] - bin[j]) ** 2)
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / ((bin[j + 2] - bin[j + 1]) ** 2)
    return fbank


'''
训练新特征之前
检查接下来使用的
binpoint
'''

file_path = myconfig.SAVED_BINPOINT_PATH
loaded_data = torch.load(file_path)
bins = loaded_data['binpoint_params']
binpoints = bins.tolist()
binpoints = np.round(binpoints)


# AAF特征提取:
# 需要手动切换
def extract_features(audio_file):
    waveform, sample_rate = sf.read(audio_file)
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())
    if sample_rate != p_sample_rate:
        waveform = librosa.resample(y=waveform, orig_sr=sample_rate, target_sr=p_sample_rate)
    enhance_signal = preemphasis(signal=waveform, coeff=p_coeff)
    framewin_signal = framesig(sig=enhance_signal, frame_len=p_sample_rate * p_winlen, frame_step=p_sample_rate * p_winstep, winfunc=np.hamming)
    spec_signal = powspec(framewin_signal, p_NFFT)
    energy = np.sum(spec_signal, 1)  # this stores the total energy in each frame
    energy = np.where(energy <= 0, np.finfo(float).eps, energy)  # if energy is zero, we get problems with log
    energy = energy[:, np.newaxis]  # 添加新轴，使 energy 的形状变为 (num_frames, 1)
    energy = energy.reshape(-1)
    # AAF
    fb = create_AAF_matrix(binpoints, p_NFFT)
    feat = np.dot(spec_signal, fb.T)
    feat[feat <= 0] = 1e-10
    feat[:, 0] = energy
    return np.log(feat)


# MFCC特征提取:
# 需要手动切换
# def extract_features(audio_file):
#     """Extract MFCC features from an audio file, shape=(TIME, MFCC)."""
#     waveform, sample_rate = sf.read(audio_file)
#     # Convert to mono-channel.
#     if len(waveform.shape) == 2:
#         waveform = librosa.to_mono(waveform.transpose())
#     # Convert to 16kHz.
#     if sample_rate != 16000:
#         waveform = librosa.resample(waveform, sample_rate, 16000)
#     features = python_speech_features.mfcc(signal=waveform,
#                                            samplerate=p_sample_rate,
#                                            winlen=p_winlen,
#                                            winstep=p_winstep,
#                                            numcep=myconfig.N_MFCC,
#                                            nfilt=myconfig.N_MFCC,
#                                            nfft=p_NFFT,
#                                            lowfreq=p_lowfreq,
#                                            highfreq=p_highfreg,
#                                            preemph=p_coeff)
#     return features


def extract_sliding_windows(features):
    """Extract sliding windows from features."""
    sliding_windows = []
    start = 0
    while start + myconfig.SEQ_LEN <= features.shape[0]:
        sliding_windows.append(features[start: start + myconfig.SEQ_LEN, :])
        start += myconfig.SLIDING_WINDOW_STEP
    return sliding_windows


def get_triplet_features(spk_to_utts):
    """Get a triplet of anchor/pos/neg features."""
    anchor_utt, pos_utt, neg_utt = dataset.get_triplet(spk_to_utts)
    return (extract_features(anchor_utt), extract_features(pos_utt), extract_features(neg_utt))


def trim_features(features, apply_specaug):
    """Trim features to SEQ_LEN."""
    full_length = features.shape[0]
    start = random.randint(0, full_length - myconfig.SEQ_LEN)
    trimmed_features = features[start: start + myconfig.SEQ_LEN, :]
    if apply_specaug:
        trimmed_features = specaug.apply_specaug(trimmed_features)
    return trimmed_features


class TrimmedTripletFeaturesFetcher:
    """The fetcher of trimmed features for multi-processing."""

    def __init__(self, spk_to_utts):
        self.spk_to_utts = spk_to_utts

    def __call__(self, _):
        """Get a triplet of trimmed anchor/pos/neg features."""
        anchor, pos, neg = get_triplet_features(self.spk_to_utts)
        while (anchor.shape[0] < myconfig.SEQ_LEN or pos.shape[0] < myconfig.SEQ_LEN or neg.shape[0] < myconfig.SEQ_LEN):
            anchor, pos, neg = get_triplet_features(self.spk_to_utts)
        return np.stack([trim_features(anchor, myconfig.SPECAUG_TRAINING), trim_features(pos, myconfig.SPECAUG_TRAINING), trim_features(neg, myconfig.SPECAUG_TRAINING)])


def get_batched_triplet_input(spk_to_utts, batch_size, pool=None):
    """Get batched triplet input for PyTorch."""
    fetcher = TrimmedTripletFeaturesFetcher(spk_to_utts)
    if pool is None:
        input_arrays = list(map(fetcher, range(batch_size)))
    else:
        input_arrays = pool.map(fetcher, range(batch_size))
    batch_input = torch.from_numpy(np.concatenate(input_arrays)).float()
    return batch_input
