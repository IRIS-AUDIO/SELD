import numpy as np
import os
import torchaudio
import torch

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def degree_to_radian(degree):
    return degree * np.pi / 180


def radian_to_degree(radian):
    return radian / np.pi * 180

def cal_rms(amp):
	# calculate root mean square
    eps = np.finfo(np.float32).eps
    return torch.sqrt(torch.mean(torch.square(amp), axis=-1) + eps)
    
def adjust_noise(noise, source, snr):
    eps = np.finfo(np.float32).eps
    noise_rms = cal_rms(noise) # noise rms

    num = cal_rms(source) # source rms
    den = np.power(10., snr/20.)
    desired_noise_rms = num/den

    # calculate gain
    try:
        gain = desired_noise_rms / (noise_rms + eps)
    except OverflowError as error:
        gain = 1.
    
    noise = torch.mul(gain, noise.T).T
    mix = source + noise
  
    return mix, source, noise, gain