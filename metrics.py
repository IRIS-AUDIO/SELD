import tensorflow as tf


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


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])

