ms = r * 0.1 # ms frame
ori_label_list = []
with open(original_label, 'r') as o: # Original metadata
    for line in o.readlines():
        # add original label to new label
        frame, cls, unknown, azi, ele = list(map(int, line.split(',')))                    
        origin_label = [frame, cls, azi, ele]
        ori_label_list.append(original_label)
ori_label_list = np.asarray([ori_label_list])

new_label_list = []
with open(mix_label, 'r') as o_2: # mixing metadata
    for line in o_2.readlines():
        frame_2, cls_2, unknown_2, azi_2, ele_2 =\
            list(map(int, o_2_line[index].strip('\n').split(',')))
        added_label = [frame_2, cls_2, azi_2, ele_2]
        new_label_list.append(added_label)
new_label_list = np.asarray([new_label_list])

for new_label in new_label_list:
    if len(np.where(ori_label_list[:,0] == new_label[0])[0]):
        frame_index = np.where(ori_label_list[:,0] == new_label[0])[0]
        if ori_label[frame_index,1] == new_label[1]:
            np.delete(ori_label, frame_index, 0)
            mix[int(ms*frame_index):int(ms*(frame_index+1))] = \
            source[int(ms*frame_index):int(ms*(frame_index+1))]                   

            