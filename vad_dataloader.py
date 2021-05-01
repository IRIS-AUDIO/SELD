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
                    n_fft=1024, n_mels=80, sr=16000, **kwargs):
    n_samples = len(wav_fnames)
    assert n_samples == len(label_fnames)

    window = preprocess_window(window)
    n_fft = tf.cast(n_fft, tf.int32)
    mel_scale = get_mel_scale(n_fft, n_mels, sr)

    def generator(wav_fnames, label_fnames, n_fft):
        for i in range(n_samples):
            yield extract_feat_label(
                wav_fnames[i], label_fnames[i],
                n_fft=n_fft, n_mels=n_mels, mel_scale=mel_scale, **kwargs)

    args = (wav_fnames, label_fnames, n_fft)
    dataset = tf.data.Dataset.from_generator(
        generator,
        args=args,
        output_signature=(
            tf.TensorSpec(shape=(None, n_mels, 1), 
                          dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)))
    dataset = dataset.cache()
    dataset = dataset.map(apply_window(window))

    return dataset


def get_vad_dataset_from_pairs(feat_label_pairs, window):
    n_mels, n_chan = feat_label_pairs[0][0].shape[1:]
    window = preprocess_window(window)

    def generator():
        for feat, label in feat_label_pairs:
            yield feat, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, n_mels, n_chan), 
                          dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)))
    dataset = dataset.cache()
    dataset = dataset.map(apply_window(window))

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


def preprocess_window(window):
    if isinstance(window, int):
        window = tf.range(window)
    window = tf.convert_to_tensor(window, dtype=tf.int32)
    window -= tf.reduce_min(window)
    return window


def apply_window(window):
    win_size = max(window)

    def _apply(feats, labels):
        n_frames = tf.shape(labels)[0]
        offset = tf.random.uniform(shape=[],
                                   maxval=n_frames-win_size,
                                   dtype=tf.int32)
        return (tf.gather(feats, window+offset), 
                tf.gather(labels, window+offset))
    return _apply


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
    '''
    WAV_PATH = '/datasets/datasets/ai_challenge/TIMIT_NOISEX_extended/TEST/WAV'
    LABEL_PATH = '/datasets/datasets/ai_challenge/TIMIT_SoundIdea/VALIDATION/LABEL'
    '''

    # TEST
    # LibriSpeech_Aurora - TEST
    WAV_PATH = '/datasets/datasets/ai_challenge/LibriSpeech_ext_Aurora/DATASET/WAV'
    LABEL_PATH = '/datasets/datasets/ai_challenge/LibriSpeech_ext_Aurora/LABEL'

    mel_scale = get_mel_scale(1024, 80, 16000)

    feats_labels = Parallel(n_jobs=8)(
        delayed(extract_feat_label)(w, l, mel_scale=mel_scale)
        for w, l in tqdm.tqdm(zip(wav_fnames, label_fnames)))

    joblib.dump(feats_labels, 'libri_aurora_test.jl')
    '''

    wav_fnames, label_fnames = extract_vad_fnames(WAV_PATH, LABEL_PATH)
    window = [-19, -10, -1, 0, 1, 10, 19]
    dataset = get_vad_dataset(wav_fnames, label_fnames, window, n_fft=1024)
    '''
    dataset = get_vad_dataset_from_pairs(
        joblib.load('timit_soundidea_train.jl'), window=window)
    '''
    dataset = data_loader(dataset, loop_time=1, batch_size=256)

    for x, y in dataset.take(2):
        print(x.shape, y.shape)

