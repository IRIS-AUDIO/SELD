import numpy as np
import torchaudio
from data_utils import *


def load_audio(filename):
    wav, rate = torchaudio.load(filename)
    return wav.numpy(), rate


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

