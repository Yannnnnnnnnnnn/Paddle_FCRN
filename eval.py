# 2021-12-3
# created by Yan
# following code from https://github.com/iro-cp/FCRN-DepthPrediction/blob/master/matlab/error_metrics.m

import numpy as np

def mean_relative_error(pred,gt):

    rel = np.abs(pred-gt)/gt

    return rel.mean()

def root_squd_error(pred,gt):

    rms = pred-gt
    rms = rms * rms

    return np.sqrt(rms.mean())

def log10_error(pred,gt):

    # log only work >0
    mask = pred>1e-6
    pred = pred[mask]
    gt = gt[mask]

    log10_error = np.abs( np.log10(pred) -np.log10(gt) )

    return log10_error.mean()

