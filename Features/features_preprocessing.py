import librosa
import soundfile as sf
import random
import torch
import numpy as np
import myconfig
import dataset
import specaug
import decimal
import math
import logging

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


# 这个函数用于统一输入尺寸
def pad_or_trim_features(features, target_length=1500):
    current_length = features.shape[0]
    feature_dim = features.shape[1]
    if current_length > target_length:
        # Trim the features to the target length
        adjusted_features = features[:target_length, :]
    elif current_length < target_length:
        # Pad the features with zeros to the target length
        padding = np.zeros((target_length - current_length, feature_dim))
        adjusted_features = np.vstack((features, padding))
    else:
        # The current length is already equal to the target length
        adjusted_features = features
    return adjusted_features


# 语音特征提取函数
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
    energy = np.where(energy == 0, np.finfo(float).eps, energy)  # if energy is zero, we get problems with log
    energy = energy[:, np.newaxis]  # 添加新轴，使 energy 的形状变为 (num_frames, 1)
    energy = pad_or_trim_features(energy)
    energy = energy.reshape(-1)
    spec_signal[spec_signal == 0] = 1e-10
    log_spec_signal = np.log(spec_signal)
    log_spec_signal_fixed = pad_or_trim_features(log_spec_signal)
    log_spec_signal_fixed[:, 0] = energy
    return log_spec_signal_fixed


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
