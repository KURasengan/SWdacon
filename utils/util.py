import os
import random
import json

from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.nn.functional as F
from typing import List, Union


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


# 시드 고정 함수
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


# RLE 디코딩 함수
def rle_decode(mask_rle: Union[str, int], shape=(224, 224)) -> np.array:
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if mask_rle == -1:
        return np.zeros(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# dice score 계산 함수
def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    """
    Calculate Dice Score between two binary masks.
    """
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (
        np.sum(prediction) + np.sum(ground_truth) + smooth
    )


def calculate_dice_scores(validation_df, img_shape=(224, 224)) -> List[float]:
    """
    Calculate Dice scores for a dataset.
    """
    # Extract the mask_rle columns
    pred_mask_rle = validation_df.iloc[:, 3]
    gt_mask_rle = validation_df.iloc[:, 4]

    def calculate_dice(pred_rle, gt_rle):
        pred_mask = rle_decode(pred_rle, img_shape)
        gt_mask = rle_decode(gt_rle, img_shape)
        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return dice_score(pred_mask, gt_mask)
        else:
            return None  # No valid masks found, return None

    dice_scores = [
        calculate_dice(pred_rle, gt_rle)
        for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    ]
    dice_scores = [
        score for score in dice_scores if score is not None
    ]  # Exclude None values
    return np.mean(dice_scores)


def calculate_nums_pixel(validation_df, img_shape=(224, 224)):
    """
    Validation의 건물 pixel 수와 Prediction의 건물 pixel 수를 계산합니다.
    더 많이 예측하는지, 덜 예측하는지 기준을 잡고 threshold를 조정에 도움이 될 수 있습니다.
    """
    eps = 1e-6
    batch_temp, count = 0, 0
    more_pred, less_pred = 0, 0
    pred_mask = validation_df.iloc[:, 2]
    gt_mask = validation_df.iloc[:, 3]
    for p_mask, t_mask in zip(pred_mask, gt_mask):
        if np.sum(rle_decode(t_mask, img_shape)):
            count += 1
            temp = float(
                int(
                    np.sum(rle_decode(t_mask, img_shape))
                    - int(np.sum(rle_decode(p_mask, img_shape)))
                )
                / (int(np.sum(rle_decode(t_mask, img_shape))) + eps)
            )
            if temp > 0:
                less_pred += 1
            elif temp < 0:
                more_pred += 1
            batch_temp += temp
    return batch_temp / count, more_pred, less_pred


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
