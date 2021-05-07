#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

file_list = sorted(glob('./DCASE2020/feat_label/foa_dev_label/*'))
sound_list = sorted(glob('/root/datasets/DCASE2020/foa_dev/*'))
os.makedirs('./single_sound', exist_ok=True)
os.makedirs('./single_label', exist_ok=True)

train = [3,4,5,6]
class_num = 14

file_list = [f for f in file_list
            if int(f[f.rfind(os.path.sep)+5]) in train] 
sound_list = [f for f in sound_list
            if int(f[f.rfind(os.path.sep)+5]) in train]

num_single_class = [0 for i in range(class_num)]

def process(inputs):
    file, sound = inputs
    data, sr = sf.read(sound)
    temp_npy = np.load(file)
    label_answer = temp_npy[:,:14]
    check_single = np.sum(label_answer, axis=1)
    single_index = np.where(check_single == 1)
    
    check_same_label = 0
    check_sequence = 0
    index_list = []
    
    frame_len = 0 # length of start frame
    start_location = 0 # start location of specific class
    new_location = 0 # check weather start frame changed
    for single in single_index[0]:        
        if new_location == 0:
            check_sequence = single - 1
            check_same_label = np.argwhere(label_answer[single] == 1)[0][0]
            start_location = single
            frame_len = 1
            new_location = 1
            
        if (single - 1) == check_sequence and \
            check_same_label == np.argwhere(label_answer[single] == 1)[0][0] :
            check_sequence = single
            frame_len += 1
            new_location = 1
            
        else:
            if frame_len >= 10 : 
                index_list.append([start_location, frame_len, check_same_label])
            new_location = 0
            
    for index in index_list:
        save_npy = np.zeros([frame_len, 4])
        num_single_class[index[2]] += 1
        sf.write('single_sound/single_' + str(num_single_class[index[2]]) +\
                 '_' + str(index[1]) +'_' + str(index[2]) + '.wav',
                 data[int(index[0]*0.1*sr):\
                      int((index[0]+index[1])*0.1*sr),:], sr)
        
        save_npy = np.copy(temp_npy[index[0]:index[0]+index[1],:])
        np.save('single_label/single_' + str(num_single_class[index[2]]) + '_'\
                + str(index[1]) +'_' + str(index[2]) + '.npy', save_npy)

with ThreadPoolExecutor() as pool:
    list(map(process, tqdm(zip(file_list, sound_list))))
    