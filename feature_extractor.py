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

    # [chan, freq, time] -> [time, freq, chan]
    outputs = torch.transpose(outputs, 0, 2).numpy()
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
    path = '/datasets/datasets/DCASE2020/foa_dev/'
    import os
    os.chdir(path)
    f = os.listdir()[0]
    w, r = torchaudio.load(f)
    spec = complex_spec(w)

    foa = foa_intensity_vectors(spec)
    print(foa.dtype, foa.shape)

    out = extract_features(f, mode='foa')
    print(out.shape, out.dtype)

