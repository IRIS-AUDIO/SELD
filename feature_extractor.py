import numpy as np
import torch
import torchaudio
from data_utils import *
from torch.fft import irfft


''' For SELDnet Data '''
def extract_features(path: str,
                     mode='foa',
                     n_mels=64,
                     **kwargs) -> np.ndarray:
    wav, r = torchaudio.load(path)
    spec = complex_spec(wav, **kwargs)

    if mode == 'foa':
        melscale = torchaudio.transforms.MelScale(n_mels=n_mels,
                                                  sample_rate=r)
        mel_spec = torchaudio.functional.complex_norm(spec, power=2.)
        mel_spec = melscale(mel_spec)
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec,
            multiplier=10.,
            amin=1e-10,
            db_multiplier=np.log10(max(1., 1e-10)), # log10(max(ref, amin))
        )

        foa = foa_intensity_vectors(spec)
        foa = melscale(foa)

        outputs = torch.cat([mel_spec, foa], axis=0)
    elif mode == 'gcc':
        raise NotImplementedError()
    else:
        raise ValueError('invalid mode')

    # [chan, freq, time] -> [time, freq, chan]
    outputs = torch.transpose(outputs, 0, 2).numpy()
    return outputs


def extract_labels(path: str, n_classes=14, max_frames=None):
    labels = []
    with open(path, 'r') as o:
        for line in o.readlines():
            frame, cls, _, azi, ele = list(map(int, line.split(',')))
            labels.append([frame, cls, azi, ele])
    labels = np.stack(labels, axis=0)

    # polar to cartesian
    labels = np.concatenate(
        [labels[..., :2], polar_to_cartesian(labels[..., 2:])], axis=-1)

    # create an empty output
    output_len = labels[..., 0].max().astype('int32') + 1
    if max_frames is not None:
        output_len = max(max_frames, output_len)
    outputs = np.zeros((output_len, 4, n_classes), dtype='float32')

    # fill in the output
    for label in labels:
        outputs[int(label[0]), :, int(label[1])] = [1., *label[2:]] 
    outputs = outputs.reshape([-1, 4*n_classes])

    return outputs


''' Feature Extraction '''
def complex_spec(wav: torch.Tensor, 
                 pad=0,
                 n_fft=512,
                 win_length=None,
                 hop_length=None,
                 normalized=False) -> torch.Tensor:
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2
    spec = torchaudio.functional.spectrogram(
        wav, 
        pad=pad, 
        window=torch.hann_window(win_length),
        n_fft=n_fft,
        hop_length=hop_length, 
        win_length=win_length, 
        power=None,
        normalized=normalized)
    return spec


def foa_intensity_vectors(complex_specs: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(complex_specs):
        complex_specs = torch.view_as_complex(complex_specs)

    # complex_specs: [chan, freq, time]
    print(complex_specs.shape, complex_specs.dtype)
    IVx = torch.real(torch.conj(complex_specs[0]) * complex_specs[3])
    IVy = torch.real(torch.conj(complex_specs[0]) * complex_specs[1])
    IVz = torch.real(torch.conj(complex_specs[0]) * complex_specs[2])

    norm = torch.sqrt(IVx**2 + IVy**2 + IVz**2)
    IVx = IVx / norm
    IVy = IVy / norm
    IVz = IVz / norm

    # apply mel matrix without db ...
    return torch.stack([IVx, IVy, IVz], axis=0)


''' Unit Conversion '''
def cartesian_to_polar(coordinates):
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)

    if coordinates.shape[-1] != 3:
        raise ValueError('only 3D cartesian coordinates are allowed')

    x = coordinates[..., 0]
    y = coordinates[..., 1]
    z = coordinates[..., 2]

    azimuth = radian_to_degree(np.arctan2(y, x))
    elevation = radian_to_degree(np.arctan2(z, np.sqrt(x**2 + y**2)))
    r = np.sqrt(x**2 + y**2 + z**2)

    return np.stack([azimuth, elevation, r], axis=-1)


def polar_to_cartesian(coordinates):
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)

    azimuth = degree_to_radian(coordinates[..., 0])
    elevation = degree_to_radian(coordinates[..., 1])
    if coordinates.shape[-1] == 3:
        r = coordinates[..., 2]
    else:
        r = 1

    x = r * np.cos(azimuth) * np.cos(elevation)
    y = r * np.sin(azimuth) * np.cos(elevation)
    z = r * np.sin(elevation)

    return np.stack([x, y, z], axis=-1)


if __name__ == '__main__':
    import os
    f = 'label.csv'
    labels = extract_labels(f, max_frames=600)
    y_target = np.load(f.replace('.csv', '.npy'))

    hop_length = int(24000 * 0.02)
    win_length = hop_length * 2
    n_fft = 2 ** (win_length-1).bit_length()
    x = extract_features(
        'x.wav', mode='foa', 
        hop_length=hop_length, win_length=win_length, n_fft=n_fft)
    x_target = np.load('x.npy').reshape(-1, 7, 64)

