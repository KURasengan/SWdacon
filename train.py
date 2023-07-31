import argparse

import os
import cv2
import pandas as pd
import numpy as np
import random
from typing import List, Union
from joblib import Parallel, delayed

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.model_selection import KFold

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import datetime
import pytz
from utils import read_json, set_seed


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="SWDacon")
    args.add_argument(
        "-c",
        "--config",
        default="base_config.json",
        type=str,
        help='Enter the path to the config file. (default: "./config.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config["trainer"]["seed"])
