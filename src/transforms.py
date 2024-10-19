# Image Transforms and Augementation
# BoMeyering 2024

import albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import ToTensorV2
from typing import Tuple, Iterable

# Set current means and std for PGC dataset
PGC_MEANS = [0.3454266254192607, 0.4017423547357385, 0.23571134798357693]
PGC_STD = [0.19282933842902986, 0.204365895781311, 0.16575982218566662]

def get_pgc_transforms(resize: Tuple=(1024, 1024), means: Iterable=PGC_MEANS, std: Iterable=PGC_STD) -> albumentations.Compose:
    """
    Return a transform function for validation transforms, i.e. just resize and normalize.

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (512, 512).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ])

    return transforms

def get_marker_transforms(resize: Tuple=(1024, 1024)) -> albumentations.Compose:
    """
    Return a transform function for marker model transforms, i.e. just resize and normalize.

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (1024, 1024).

    Returns:
        albumentations.Compose: A Compose function to use in the datasets.
    """
    transforms = A.Compose([
            A.Resize(*resize, p=1),
            A.Normalize(),
            ToTensorV2(p=1)
    ])
    
    return transforms
