#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow_io as tfio
import tensorflow as tf


def foa_intensity_vectors_tf(spectrogram, eps=1e-8):

    # complex_specs: [chan, freq, time]
    conj_zero = tf.math.conj(spectrogram[0])
    IVx = tf.math.real(conj_zero * spectrogram[3])
    IVy = tf.math.real(conj_zero * spectrogram[1])
    IVz = tf.math.real(conj_zero * spectrogram[2])

    norm = tf.math.sqrt(IVx**2 + IVy**2 + IVz**2)
    norm = tf.math.maximum(norm, tf.zeros_like(norm)+eps)
    IVx = IVx / norm
    IVy = IVy / norm
    IVz = IVz / norm

    # apply mel matrix without db ...
    return tf.stack([IVx, IVy, IVz], axis=0)


def gcc_features_tf(complex_specs, n_mels):
    n_chan = complex_specs.shape[0]
    gcc_feat = []
    for m in range(n_chan):
        for n in range(m+1, n_chan):
            R = tf.math.conj(complex_specs[m]) * complex_specs[n]
            print(R.shape)
            cc = tf.signal.irfft(tf.math.exp(1.j*tf.complex(tf.math.angle(R),0.0)))
            cc = tf.concat([cc[-n_mels//2:], cc[:(n_mels+1)//2]], axis=0)
            gcc_feat.append(cc)

    return tf.stack(gcc_feat, axis=0)


def get_preprocessed_x_tf(wav, sr, mode='foa', n_mels=64,
                          multiplier=5, max_label_length=600, win_length=960,
                          hop_length=480, n_fft=1024):

    mel_mat = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=n_mels,
                                                    num_spectrogram_bins=n_fft//2+1,
                                                    sample_rate=sr,
                                                    lower_edge_hertz=0,
                                                    upper_edge_hertz=sr//2)


    audio_tensor = tf.cast(audio.to_tensor(), tf.float32) / 32768.0

    audio_tensor = tf.transpose(audio_tensor)
    spectrogram = tf.signal.stft(audio_tensor, 960, 480, 1024, pad_end=True)
    norm_spec = tf.math.abs(spectrogram)
    mel_spec = tf.matmul(norm_spec, mel_mat)
    mel_spec = tfio.experimental.audio.dbscale(mel_spec, top_db=80)
    features = [mel_spec]
        
    if mode == 'foa':
        foa = foa_intensity_vectors_tf(spectrogram)
        foa = tf.matmul(foa, mel_mat)
        foa = tfio.experimental.audio.dbscale(foa, top_db=80)
        features.append(foa)
        
    elif mode == 'mic':
        gcc = gcc_features_tf(spectrogram, n_mels=n_mels)
        features.append(gcc)
    
    else:
        raise ValueError('invalid mode')
    
    features = tf.concat(features, axis=0)
    features = tf.transpose(features, perm=[1, 2, 0])
    
    cur_len = features.shape[0]
    max_len = max_label_length * multiplier
    
    if cur_len < max_len: 
        pad = tf.constant([[0, max_len-cur_len], [0,0], [0,0]])
        features = tf.pad(features, pad, 'constant')
    else:
        features = features[:max_len]
    return features

if __name__ == "__main__":
    audio = tfio.audio.AudioIOTensor('/home/pjh/seld-dcase2020/foa_dev/fold1_room1_mix001_ov1.wav')
    get_preprocessed_x_tf(audio, 24000)