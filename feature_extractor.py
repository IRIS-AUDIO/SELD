import numpy as np
import os
import torch
import torchaudio
import tqdm
from functools import partial
from glob import glob
from torch.fft import irfft

from data_utils import *
from utils import get_device
import random as rnd


''' For SELDnet Data '''
def extract_seldnet_data(feature_path: str,
                         feature_output_path: str,
                         label_path: str,
                         label_output_path: str,
                         mode='foa',
                         use_aug = False,
                         **kwargs):

    if feature_output_path == label_output_path:
        raise ValueError('output folders for features and labels must differ')

    f_paths = sorted(glob(os.path.join(feature_path, '*.wav')))
    l_paths = sorted(glob(os.path.join(label_path, '*.csv')))

    if len(f_paths) != len(l_paths):
        raise ValueError('# of features and labels are not matched')

    create_folder(feature_output_path)
    create_folder(label_output_path)

    def extract_name(path):
        return path[path.rfind(os.path.sep)+1:path.rfind('.')]

    for f, l in tqdm.tqdm(zip(f_paths, l_paths)):
        # name must match
        name = extract_name(f)
        if name != extract_name(l):
            raise ValueError('feature, label must share the same name')
        # RANDOMLY PICK ANOTHER WAV FILE TO MIX       


        # Randomly Using augment or not
        if use_aug:
            removed_f = f_paths.copy() 
            removed_f.remove(f)

            removed_l = l_paths.copy()
            removed_l.remove(l)
            #random_number is mixed index
            random_number = rnd.randrange(len(f_paths) -1)
            pick_or = rnd.randrange(0,3)
            if pick_or == 1:
                # file and label that will be mixed
                mixed_f = removed_f[random_number]
                mixed_l = removed_l[random_number]
                # mix and extract label
                mixed_f, mixed_l = mix_and_extract(f, mixed_f, l, mixed_l)
                mixed_f, mixed_l = preprocess_features_labels(mixed_f, mixed_l)
            
        f = extract_features(f, mode=mode, **kwargs)
        l = extract_labels(l)
        f, l = preprocess_features_labels(f, l)
        
        new_name = name + '.npy'
        np.save(os.path.join(feature_output_path, new_name), f)
        np.save(os.path.join(label_output_path, new_name), l)
        
        if use_aug == True:
            if pick_or == 1:        
                #save file when we need augmentation
                aug_new_name = name + '_aug.npy'
                np.save(os.path.join(feature_output_path, aug_new_name), mixed_f)
                np.save(os.path.join(label_output_path, aug_new_name), mixed_l)

def extract_features(path: str,
                     mode='foa',
                     n_mels=64,
                     **kwargs) -> np.ndarray:
    device = get_device()
    wav, r = torchaudio.load(path)
    melscale = torchaudio.transforms.MelScale(n_mels=n_mels,
                                              sample_rate=r).to(device)
    spec = complex_spec(wav.to(device), **kwargs)

    mel_spec = torchaudio.functional.complex_norm(spec, power=2.)
    mel_spec = melscale(mel_spec)
    mel_spec = torchaudio.functional.amplitude_to_DB(
        mel_spec,
        multiplier=10.,
        amin=1e-10,
        db_multiplier=np.log10(max(1., 1e-10)), # log10(max(ref, amin))
        top_db=80.,
    )

    features = [mel_spec]
    if mode == 'foa':
        foa = foa_intensity_vectors(spec)
        foa = melscale(foa)
        features.append(foa)
    elif mode == 'mic':
        gcc = gcc_features(spec, n_mels=n_mels)
        features.append(gcc)
    else:
        raise ValueError('invalid mode')

    features = torch.cat(features, axis=0)

    # [chan, freq, time] -> [time, freq, chan]
    features = torch.transpose(features, 0, 2).cpu().numpy()
    return features


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


def preprocess_features_labels(features: np.ndarray, 
                               labels: np.ndarray, 
                               max_label_length=600, 
                               multiplier=5):
    '''
    INPUT
    features: [time_f, freq, chan] shaped sample
    labels:   [time_l, 4*n_classes] shaped sample
    max_label_length: length of labels (time) will be set to given value
    multiplier: how many feature frames are related to a single label frame

    OUTPUT
    features: [max_label_length*multiplier, freq, chan]
    labels: [max_label_length, 4*n_classes]
    '''
    cur_len = labels.shape[0]
    max_len = max_label_length

    if cur_len < max_len: 
        labels = np.pad(labels, ((0, max_len-cur_len), (0,0)), 'constant')
    else:
        labels = labels[:max_len]

    cur_len = features.shape[0]
    max_len = max_label_length * multiplier
    if cur_len < max_len: 
        features = np.pad(features, 
                          ((0, max_len-cur_len), (0,0), (0,0)),
                          'constant')
    else:
        features = features[:max_len]

    return features, labels


''' Feature Extraction '''
def complex_spec(wav: torch.Tensor, 
                 pad=0,
                 n_fft=512,
                 win_length=None,
                 hop_length=None,
                 normalized=False) -> torch.Tensor:
    device = get_device()
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2
    spec = torchaudio.functional.spectrogram(
        wav, 
        pad=pad, 
        window=torch.hann_window(win_length, device=device),
        n_fft=n_fft,
        hop_length=hop_length, 
        win_length=win_length, 
        power=None,
        normalized=normalized)
    return spec


def foa_intensity_vectors(complex_specs: torch.Tensor, 
                          eps=1e-8) -> torch.Tensor:
    if not torch.is_complex(complex_specs):
        complex_specs = torch.view_as_complex(complex_specs)

    # complex_specs: [chan, freq, time]
    IVx = torch.real(torch.conj(complex_specs[0]) * complex_specs[3])
    IVy = torch.real(torch.conj(complex_specs[0]) * complex_specs[1])
    IVz = torch.real(torch.conj(complex_specs[0]) * complex_specs[2])

    norm = torch.sqrt(IVx**2 + IVy**2 + IVz**2)
    norm = torch.maximum(norm, torch.zeros_like(norm)+eps)
    IVx = IVx / norm
    IVy = IVy / norm
    IVz = IVz / norm

    # apply mel matrix without db ...
    return torch.stack([IVx, IVy, IVz], axis=0)


def gcc_features(complex_specs: torch.Tensor,
                 n_mels: int) -> torch.Tensor:
    if not torch.is_complex(complex_specs):
        complex_specs = torch.view_as_complex(complex_specs)

    # based on the codes from DCASE2020 SELDnet cls_feature_class.py
    # complex_specs: [chan, freq, time]
    n_chan = complex_specs.size(0)
    gcc_chan = n_chan * (n_chan - 1) // 2

    gcc_feat = []
    for m in range(n_chan):
        for n in range(m+1, n_chan):
            R = torch.conj(complex_specs[m]) * complex_specs[n]
            cc = torch.fft.irfft(torch.exp(1.j*torch.angle(R)), dim=0)
            cc = torch.cat([cc[-n_mels//2:], cc[:(n_mels+1)//2]], axis=0)
            gcc_feat.append(cc)

    return torch.stack(gcc_feat, axis=0)


''' Normalizating Features '''
def calculate_statistics(feature_path: str):
    features = sorted(glob(os.path.join(feature_path, '*.npy')))
    features = np.concatenate([np.load(f) for f in features], 0)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return mean, std


def apply_normalizer(feature_path, new_feature_path, mean, std, eps=1e-8):
    features = sorted(glob(os.path.join(feature_path, '*.npy')))
    create_folder(new_feature_path)

    for feature in tqdm.tqdm(features):
        new_name = os.path.join(new_feature_path, 
                                os.path.split(feature)[1])
        new_feat = (np.load(feature) - mean) / np.maximum(std, eps)
        np.save(new_name, new_feat)


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


def mix_and_extract(original, mix, original_label, mix_label, 
                    n_classes = 14, max_frames = None, mode='foa', n_mels=64,
                    **kwargs):
    
    #load original and random sound
    original_sound, r = torchaudio.load(original)
    mixing_sound, r = torchaudio.load(mix)
    ms = r * 0.1 # ms frame
    snr_list = [-10 + 2*i for i in range(11)]
    snr = rnd.sample(snr_list, 1)[0]
    
    # mix two sound
    mix, source, noise, _ = adjust_noise(mixing_sound, original_sound, snr)
    ori_label_list = []
    with open(original_label, 'r') as o: # Original metadata
        for line in o.readlines():
            # add original label to new label
            frame, cls, unknown, azi, ele = list(map(int, line.split(',')))                    
            origin_label = [frame, cls, azi, ele]
            ori_label_list.append(origin_label)
    ori_label_list = np.asarray([ori_label_list])
    ori_label_list = ori_label_list.reshape(-1,4)

    new_label_list = []
    with open(mix_label, 'r') as o_2: # mixing metadata
        for o_2_line in o_2.readlines():
            frame_2, cls_2, unknown_2, azi_2, ele_2 =\
                list(map(int, o_2_line.strip('\n').split(',')))
            added_label = [frame_2, cls_2, azi_2, ele_2]
            new_label_list.append(added_label)
    new_label_list = np.asarray([new_label_list])
    new_label_list = new_label_list.reshape(-1,4)
    for new_label in new_label_list:
        if len(np.where(ori_label_list[:,0] == new_label[0])[0]):
            frame_indexes = np.where(ori_label_list[:,0] == new_label[0])[0]
            for frame_index in frame_indexes:
                if ori_label_list[frame_index,1] == new_label[1]:
                    np.delete(ori_label_list, frame_index, 0)
                    mix[int(ms*frame_index):int(ms*(frame_index+1))] = \
                    source[int(ms*frame_index):int(ms*(frame_index+1))]    
    
    # reshape metadata
    new_label = new_label.reshape(int(len(new_label)//4),-1)
    new_label = new_label[new_label[:,0].argsort()]
    
    device = get_device()
    # feature extracting
    melscale = torchaudio.transforms.MelScale(n_mels=n_mels,
                                              sample_rate=r).to(device)
    spec = complex_spec(mix.to(device), **kwargs)

    mel_spec = torchaudio.functional.complex_norm(spec, power=2.)
    mel_spec = melscale(mel_spec)
    mel_spec = torchaudio.functional.amplitude_to_DB(
        mel_spec,
        multiplier=10.,
        amin=1e-10,
        db_multiplier=np.log10(max(1., 1e-10)), # log10(max(ref, amin))
        top_db=80.,
    )

    features = [mel_spec]
    if mode == 'foa':
        foa = foa_intensity_vectors(spec)
        foa = melscale(foa)
        features.append(foa)
    elif mode == 'mic':
        gcc = gcc_features(spec, n_mels=n_mels)
        features.append(gcc)
    else:
        raise ValueError('invalid mode')

    features = torch.cat(features, axis=0)

    # [chan, freq, time] -> [time, freq, chan]
    features = torch.transpose(features, 0, 2).cpu().numpy()
    
    # label extracting
    labels = np.concatenate(
        [new_label[..., :2], polar_to_cartesian(new_label[..., 2:])], axis=-1)

    # create an empty output
    output_len = labels[..., 0].max().astype('int32') + 1
    if max_frames is not None:
        output_len = max(max_frames, output_len)
    outputs = np.zeros((output_len, 4, n_classes), dtype='float32')

    # fill in the output
    for label in labels:
        outputs[int(label[0]), :, int(label[1])] = [1., *label[2:]] 
    outputs = outputs.reshape([-1, 4*n_classes])

    return features, outputs

if __name__ == '__main__':
    # How to use
    # Extracting Features and Labels
    abspath = '/media/data1/datasets/DCASE2020' if os.path.exists('/media/data1/datasets') else '/root/datasets/DCASE2020'
    abspath = '/home/pjh/seld-dcase2020'
    FEATURE_PATH = os.path.join(abspath, 'foa_dev')
    LABEL_PATH = os.path.join(abspath, 'metadata_dev')

    # should 
    FEATURE_OUTPUT_PATH = 'foa_dev'
    LABEL_OUTPUT_PATH = 'foa_dev_label'
    NORM_FEATURE_PATH = 'foa_dev_norm'
    use_aug = False
    extract_seldnet_data(FEATURE_PATH, 
                         FEATURE_OUTPUT_PATH,
                         LABEL_PATH, 
                         LABEL_OUTPUT_PATH,
                         mode='foa', 
                         win_length=960,
                         hop_length=480,
                         n_fft=1024,
                         use_aug = use_aug)

    # Normalizing Extracted Features
    mean, std = calculate_statistics(FEATURE_OUTPUT_PATH)

    apply_normalizer(FEATURE_OUTPUT_PATH, NORM_FEATURE_PATH, mean, std)
