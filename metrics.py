import numpy as np # will remove later
import tensorflow as tf

from data_utils import radian_to_degree


def early_stopping_metric(sed_error, doa_error):
    """
    Compute early stopping metric from sed and doa errors.

    :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
    :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
    :return: seld metric result
    """
    error_rate = sed_error[0]
    f_score = sed_error[1]
    doa_error = doa_error[0] / 180 # in degrees
    recall = doa_error[1]

    return (error_rate + 1 - f_score + doa_error + 1 - recall)/4


def reshape_3Dto2D(tensor):
    return tf.reshape(tensor, (-1, tensor.shape[-1]))


class SELDMetrics:
    def __init__(self, doa_threshold=20, nb_classes=11):
        self.doa_threshold = doa_threshold
        self.class_num = nb_classes
        self.reset()

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        self.S = 0
        self.D = 0
        self.I = 0

        self.Nref = 0
        self.Nsys = 0

        self.total_DE = 0
        self.DE_TP = 0

    def compute_seld_scores(self):
        # Location-senstive detection performance
        ER = (self.S + self.D + self.I) / float(self.Nref + eps)

        prec = float(self.TP) / float(self.Nsys + eps)
        recall = float(self.TP) / float(self.Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        # Class-sensitive localization performance
        if self.DE_TP:
            DE = self.total_DE / float(self.DE_TP + eps)
        else:
            DE = 180 # When the total number of prediction is zero

        DE_prec = float(self.DE_TP) / float(self.Nsys + eps)
        DE_recall = float(self.DE_TP) / float(self.Nref + eps)
        DE_F = 2 * DE_prec * DE_recall / (DE_prec + DE_recall + eps)

        return ER, F, DE, DE_F

    def update_seld_scores_xyz(self, pred, gt):
        # blockwise
        for block_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0

            # classwise
            for class_cnt in range(self.class_num):
                if class_cnt in gt[block_cnt]:
                    self.Nref += 1
                if class_cnt in pred[block_cnt]:
                    self.Nsys += 1

                # both in GT and PRED
                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    total_spatial_dist = 0
                    total_framewise_matching_doa = 0
                    gt_ind_list = gt[block_cnt][class_cnt][0][0] # frames
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]

                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list: # gt frame is in pred
                            total_framewise_matching_doa += 1
                            pred_ind = pred_ind_list.index(gt_val)

                            # find xyz
                            gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_ind])
                            pred_arr = np.array(pred[block_cnt][class_cnt][0][1][pred_ind])

                            total_spatial_dist += distance_between_cartesian_coordinates(
                                gt_arr[0], pred_arr[0])

                    if total_spatial_dist == 0 and total_framewise_matching_doa == 0:
                        loc_FN += 1
                        self.FN += 1
                    else:
                        avg_spatial_dist = (total_spatial_dist / total_framewise_matching_doa)

                        self.total_DE += avg_spatial_dist
                        self.DE_TP += 1

                        if avg_spatial_dist <= self.doa_threshold:
                            self.TP += 1
                        else:
                            loc_FN += 1
                            self.FN += 1
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += 1
                    self.FN += 1
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += 1
                    self.FP += 1
                elif class_cnt not in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # True negative
                    self.TN += 1

            self.S += np.minimum(loc_FP, loc_FN)
            self.D += np.maximum(0, loc_FN - loc_FP)
            self.I += np.maximum(0, loc_FP - loc_FN)
        return


def distance_between_cartesian_coordinates(xyz0, xyz1): # x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # xyz0 = tf.stack([x1, y1, z1], axis=-1) # [samples, 3]
    xyz0 = tf.math.l2_normalize(xyz0)
    # xyz1 = tf.stack([x2, y2, z2], axis=-1) # [samples, 3]
    xyz1 = tf.math.l2_normalize(xyz1)

    distance = tf.reduce_sum(xyz0 * xyz1, axis=-1)
    distance = tf.clip_by_value(distance, -1, 1)
    distance = radian_to_degree(tf.math.acos(distance))
    
    return distance


def regression_label_format_to_output_format(preds):
    """
    Converts the sed (classification) and doa labels predicted 
    in regression format to dcase output format.

    :param preds: (sed, doa) prediction [nb_frames, nb_classes], [nb_frames, 3*nb_classes]
    :return: _output_dict: returns a dict containing dcase output format
    """
    sed_labels = preds[0]
    doa_labels = preds[1]

    n_frames, n_classes = sed_labels.shape
    doa_labels = tf.reshape(doa_labels, (-1, 3, n_classes))

    output_dict = {}
    for i in range(n_frames):
        classes = tf.reshape(tf.where(sed_labels[i]), (-1,))
        if len(classes):
            output_dict[i] = []
            for cls in classes:
                output_dict[i].append([cls, *doa_labels[i, :, cls]])
    return output_dict


def segment_labels(_pred_dict, _max_frames, fs=24000, label_len_s=0.1):
    '''
    Collects class-wise sound event location information 
    in segments of length 1s from reference dataset

    :param _pred_dict: Dictionary containing frame-wise sound event time 
                       and location information. Output of SELD method
    :param _max_frames: Total number of frames in the recording
    :return: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
    '''
    _nb_label_frames_1s = int(fs / float(int(fs * label_len_s)))

    nb_blocks = int(np.ceil(_max_frames/float(_nb_label_frames_1s)))
    output_dict = {x: {} for x in range(nb_blocks)}
    for frame_cnt in range(0, _max_frames, _nb_label_frames_1s):

        # Collect class-wise information for each block
        # [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        block_cnt = frame_cnt // _nb_label_frames_1s
        loc_dict = {}
        for audio_frame in range(frame_cnt, frame_cnt+_nb_label_frames_1s):
            if audio_frame not in _pred_dict:
                continue
            for value in _pred_dict[audio_frame]:
                if value[0] not in loc_dict: # value[0] = class
                    loc_dict[value[0]] = {}

                block_frame = audio_frame - frame_cnt
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append(value[1:]) # append xyz

        # Update the block wise details collected above in a global structure
        for class_cnt in loc_dict:
            if class_cnt not in output_dict[block_cnt]:
                output_dict[block_cnt][class_cnt] = []

            keys = [k for k in loc_dict[class_cnt]]
            values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

            output_dict[block_cnt][class_cnt].append([keys, values])

    '''
    output_dict[block_count][class] = [frames, xyzs]
    '''
    return output_dict

