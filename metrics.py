import tensorflow as tf

from data_utils import radian_to_degree
from utils import safe_div


class SELDMetrics:
    def __init__(self, doa_threshold=20, block_size=10):
        self.doa_threshold = doa_threshold
        self.block_size = block_size
        self.reset_states()

    def reset_states(self):
        self.TP = tf.zeros([], tf.float32)
        self.FP = tf.zeros([], tf.float32)
        self.TN = tf.zeros([], tf.float32)
        self.FN = tf.zeros([], tf.float32)

        self.S = tf.zeros([], tf.float32)
        self.D = tf.zeros([], tf.float32)
        self.I = tf.zeros([], tf.float32)

        self.Nref = tf.zeros([], tf.float32)
        self.Nsys = tf.zeros([], tf.float32)

        self.total_DE = tf.zeros([], tf.float32)
        self.DE_TP = tf.zeros([], tf.float32)

    def result(self):
        # Location-senstive detection performance
        ER = safe_div(self.S + self.D + self.I, self.Nref)

        prec = safe_div(self.TP, self.Nsys)
        recall = safe_div(self.TP, self.Nref)
        F = safe_div(2 * prec * recall, prec + recall)

        # Class-sensitive localization performance
        if self.DE_TP > 0:
            DE = safe_div(self.total_DE, self.DE_TP)
        else:
            # When the total number of prediction is zero
            DE = tf.ones([], tf.float32) * 180 

        DE_prec = safe_div(self.DE_TP, self.Nsys)
        DE_recall = safe_div(self.DE_TP, self.Nref)
        DE_F = safe_div(2 * DE_prec * DE_recall, DE_prec + DE_recall)

        return ER, F, DE, DE_F

    def update_states(self, y_true, y_pred): 
        y_true_blocks = self.split(y_true)
        y_pred_blocks = self.split(y_pred)

        for true_block, pred_block in zip(y_true_blocks, y_pred_blocks):
            self.update_block_states(true_block, pred_block)

    def split(self, labels):
        sed, doa = labels
        blocks = []
        for i in range((sed.shape[1]+self.block_size-1)//self.block_size):
            blocks.append(
                [sed[:, i*self.block_size:(i+1)*self.block_size],
                 doa[:, i*self.block_size:(i+1)*self.block_size]])

        return blocks

    def update_block_states(self, y_true_block, y_pred_block):
        sed_true, doa_true = y_true_block
        sed_pred, doa_pred = y_pred_block
        sed_pred = tf.cast(sed_pred > 0.5, sed_pred.dtype)

        # change doa shape from [..., n_classes*3] to [..., n_classes, 3]
        doa_true = tf.reshape(doa_true, (*doa_true.shape[:-1], 3, -1))
        doa_pred = tf.reshape(doa_pred, (*doa_pred.shape[:-1], 3, -1))
        perm = [*range(doa_true.ndim-2), doa_true.ndim-1, doa_true.ndim-2]
        doa_true = tf.transpose(doa_true, perm=perm)
        doa_pred = tf.transpose(doa_pred, perm=perm)

        # whether a particular class exists in a block
        # true_classes, pred_classes: [..., n_frames, n_classes] shaped Tensor
        true_classes = tf.math.reduce_max(sed_true, axis=-2, keepdims=True)
        pred_classes = tf.math.reduce_max(sed_pred, axis=-2, keepdims=True)

        self.Nref += tf.math.reduce_sum(true_classes)
        self.Nsys += tf.math.reduce_sum(pred_classes)
        self.TN += tf.math.reduce_sum((1-true_classes)*(1-pred_classes))

        false_negative = true_classes * (1-pred_classes)
        false_positive = (1-true_classes) * pred_classes

        self.FN += tf.math.reduce_sum(false_negative)
        self.FP += tf.math.reduce_sum(false_positive)
        loc_FN = tf.math.reduce_sum(false_negative, axis=(-2, -1))
        loc_FP = tf.math.reduce_sum(false_positive, axis=(-2, -1))

        ''' when a class exists in both y_true and y_pred '''
        true_positives = true_classes * pred_classes
        frames_true = sed_true * true_positives
        frames_pred = sed_pred * true_positives
        frames_matched = frames_true * frames_pred

        # [..., 1, n_classes]
        total_matched_frames = tf.reduce_sum(
            frames_matched, axis=-2, keepdims=True)
        matched_frames_exist = tf.cast(total_matched_frames > 0,
                                       total_matched_frames.dtype)
        self.DE_TP += tf.math.reduce_sum(matched_frames_exist)

        false_negative = true_positives * (1-matched_frames_exist)
        self.FN += tf.math.reduce_sum(false_negative)
        loc_FN += tf.math.reduce_sum(false_negative, axis=(-2, -1))

        # [..., n_frames, n_classes]
        angular_distances = distance_between_cartesian_coordinates(
            doa_true * tf.expand_dims(frames_matched, -1),
            doa_pred * tf.expand_dims(frames_matched, -1))
        average_distances = safe_div(
            tf.reduce_sum(angular_distances, -2, keepdims=True),
            total_matched_frames)
        self.total_DE += tf.reduce_sum(average_distances)

        close_angles = tf.cast(average_distances <= self.doa_threshold, 
                               average_distances.dtype)
        self.TP += tf.reduce_sum(close_angles * matched_frames_exist)

        false_negative = (1-close_angles) * matched_frames_exist
        self.FN += tf.reduce_sum(false_negative)
        loc_FN += tf.reduce_sum(false_negative, axis=(-2, -1))

        self.S += tf.reduce_sum(tf.math.minimum(loc_FP, loc_FN))
        self.D += tf.reduce_sum(tf.math.maximum(0, loc_FN - loc_FP))
        self.I += tf.reduce_sum(tf.math.maximum(0, loc_FP - loc_FN))


def calculate_seld_score(metric_values):
    """
    Compute early stopping metric from sed and doa errors.

    :param metric_values: [error rate (0 to 1 range), 
                           f score (0 to 1 range),
                           doa error (in degrees), 
                           frame recall (0 to 1 range)]
    :return: seld metric result
    """
    error_rate, f_score, doa_error, recall = metric_values
    doa_error = doa_error / 180 # degress to [0, 1]

    return (error_rate + 1 - f_score + doa_error + 1 - recall)/4


def distance_between_cartesian_coordinates(xyz0, xyz1):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    xyz0 = tf.math.l2_normalize(xyz0, axis=-1)
    xyz1 = tf.math.l2_normalize(xyz1, axis=-1)
    zeros = tf.cast(tf.math.reduce_sum(xyz0, axis=-1)==0, xyz0.dtype) \
          * tf.cast(tf.math.reduce_sum(xyz1, axis=-1)==0, xyz1.dtype) 

    distance = tf.reduce_sum(xyz0 * xyz1, axis=-1)
    distance = tf.clip_by_value(distance, -1, 1)
    distance = radian_to_degree(tf.math.acos(distance)) * (1-zeros)
    
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

