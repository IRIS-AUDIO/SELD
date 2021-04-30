import glob
import numpy as np
import os
import tensorflow as tf
import tqdm

import joblib
from joblib import Parallel, delayed


def extract_vad_fnames(wav_folder, label_folder):
    wav_fnames = sorted(search_sub_dirs(wav_folder))
    label_fnames = [os.path.join(label_folder, 
                                 os.path.split(fname)[1].replace('wav', 'npy'))
                    for fname in wav_fnames]
    return wav_fnames, label_fnames


def extract_feat_label(wav_fname, label_fname, **kwargs):
    wav = load_wav(wav_fname, **kwargs)
    label = load_label(label_fname, **kwargs)
    assert len(wav) == len(label)    
    return wav, label


def get_vad_dataset(wav_fnames, label_fnames, window, 
                    samples_per_wav, shuffle_wavs=False,
                    n_fft=1024, n_mels=80, sr=16000, **kwargs):
    n_samples = len(wav_fnames)
    assert n_samples == len(label_fnames)
    
    if isinstance(window, int):
        window = tf.range(window)
    window = tf.convert_to_tensor(window, dtype=tf.int32)
    window -= tf.reduce_min(window)

    n_fft = tf.cast(n_fft, tf.int32)
    mel_scale = get_mel_scale(n_fft, n_mels, sr)

    def window_generator(wav_fnames, label_fnames, n_fft):
        # shuffle fnames
        if shuffle_wavs:
            order = np.random.permutation(len(wav_fnames))
            wav_fnames = np.array(wav_fnames)[order].tolist()
            label_fnames = np.array(label_fnames)[order].tolist()

        for i in range(n_samples):
            wav, label = extract_feat_label(
                wav_fnames[i], label_fnames[i],
                n_fft=n_fft, n_mels=n_mels, mel_scale=mel_scale, **kwargs)

            n_frames = len(label)
            win_size = max(window)

            for _ in range(samples_per_wav):
                offset = tf.random.uniform(shape=[], 
                                           maxval=n_frames-win_size,
                                           dtype=tf.int32)
                yield tf.gather(wav, window+offset), \
                      tf.gather(label, window+offset)

    args = (wav_fnames, label_fnames, n_fft)
    dataset = tf.data.Dataset.from_generator(
        window_generator,
        args=args,
        output_signature=(
            tf.TensorSpec(shape=(len(window), n_mels, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(len(window),), dtype=tf.float32)))

    return dataset


def load_wav(name, n_fft=1024, n_mels=80, 
             mel_scale=None, logmel=True, normalize=True):
    wav, sr = tf.audio.decode_wav(tf.io.read_file(name))
    wav = tf.transpose(wav, (1, 0)) # to [chan, frames]

    n_fft = tf.cast(n_fft, tf.int32)
    if mel_scale is None:
        mel_scale = get_mel_scale(n_fft, n_mels, sr)

    spec = tf.signal.stft(wav, frame_length=n_fft, frame_step=n_fft//2)
    spec = tf.abs(spec)
    spec = tf.matmul(spec, mel_scale)
    spec = tf.transpose(spec, (1, 2, 0))
    
    if logmel:
        spec = tf.math.log(tf.clip_by_value(spec, 1e-8, tf.reduce_max(spec)))

    if normalize:
        min_value = tf.reduce_min(spec)
        max_value = tf.reduce_max(spec)
        spec = (spec - min_value) / (max_value - min_value)
    return spec


def load_label(name, n_fft=1024, **kwargs):
    label = tf.convert_to_tensor(np.load(name), dtype=tf.float32)
    label = tf.nn.avg_pool1d(tf.reshape(label, (1, -1, 1)),
                             n_fft,
                             n_fft//2, 
                             padding='VALID')
    return tf.squeeze(tf.round(label))


def get_mel_scale(n_fft, n_mels, sr):
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft//2+1,
        sample_rate=sr,
        lower_edge_hertz=0,
        upper_edge_hertz=sr//2)


def search_sub_dirs(path, ext='wav'):
    fnames = glob.glob(os.path.join(path, f"*.{ext}"))

    sub_dirs = os.listdir(path)
    for sd in sub_dirs:
        sub_path = os.path.join(path, sd)
        if os.path.isdir(sub_path):
            fnames += search_sub_dirs(sub_path, ext)
    return fnames


if __name__ == "__main__":
    from data_loader import data_loader

    '''
    # HOW TO USE
    # TIMIT_SoundIdea - TRAIN
    # WAV_PATH = '/datasets/datasets/ai_challenge/TIMIT_SoundIdea/TRAIN/WAV'
    # LABEL_PATH = '/datasets/datasets/ai_challenge/TIMIT_SoundIdea/TRAIN/LABEL'
    
    # TIMIT_SoundIdea - VAL
    WAV_PATH = '/datasets/datasets/ai_challenge/TIMIT_SoundIdea/VALIDATION/WAV'
    LABEL_PATH = '/datasets/datasets/ai_challenge/TIMIT_SoundIdea/VALIDATION/LABEL'

    # TIMIT_NoiseX92 - TRAIN
    WAV_PATH = '/datasets/datasets/ai_challenge/TIMIT_NOISEX_extended/TRAIN/WAV'
    LABEL_PATH = '/datasets/datasets/ai_challenge/TIMIT_SoundIdea/TRAIN/LABEL'

    # VAL
    # TIMIT_NoiseX92 - VAL
    WAV_PATH = '/datasets/datasets/ai_challenge/TIMIT_NOISEX_extended/TEST/WAV'
    LABEL_PATH = '/datasets/datasets/ai_challenge/TIMIT_SoundIdea/VALIDATION/LABEL'

    '''
    # TEST
    # LibriSpeech_Aurora - TEST
    WAV_PATH = '/datasets/datasets/ai_challenge/LibriSpeech_ext_Aurora/DATASET/WAV'
    LABEL_PATH = '/datasets/datasets/ai_challenge/LibriSpeech_ext_Aurora/LABEL'

    wav_fnames, label_fnames = extract_vad_fnames(WAV_PATH, LABEL_PATH)
    for fname in wav_fnames + label_fnames:
        assert os.path.exists(fname)

    mel_scale = get_mel_scale(1024, 80, 16000)

    feats_labels = Parallel(n_jobs=8)(
        delayed(extract_feat_label)(w, l, mel_scale=mel_scale)
        for w, l in tqdm.tqdm(zip(wav_fnames, label_fnames)))

    joblib.dump(feats_labels, 'libri_aurora_test.jl')

    '''
    window = [-19, -10, -1, 0, 1, 10, 19]
    vad_dataset = get_vad_dataset(wav_fnames, label_fnames, window, 
                                  samples_per_wav=32, shuffle_wavs=True, 
                                  n_fft=1024)

    dataset = data_loader(vad_dataset, loop_time=1, batch_size=256)
    for x, y in dataset.take(2):
        print(x.shape, y.shape)
    print(len(wav_fnames))
    '''

