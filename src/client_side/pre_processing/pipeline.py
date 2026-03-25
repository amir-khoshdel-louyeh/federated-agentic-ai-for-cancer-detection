"""
Preprocessing pipeline for HAM10000 and ISIC2019 datasets.
Includes:
- Data integrity split (lesion_id)
- DullRazor hair removal
- Shades of Gray color constancy
- Center-crop and resize
- Metadata encoding for ISIC
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from typing import Tuple, List, Dict

# 1. Data Integrity: Split by lesion_id
def split_by_lesion_id(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    lesion_ids = df['lesion_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(lesion_ids)
    split_idx = int(len(lesion_ids) * (1 - test_size))
    train_lesions = set(lesion_ids[:split_idx])
    test_lesions = set(lesion_ids[split_idx:])
    train_df = df[df['lesion_id'].isin(train_lesions)]
    test_df = df[df['lesion_id'].isin(test_lesions)]
    return train_df, test_df

# 2. DullRazor: Hair removal (Black-hat + Inpainting)
def dull_razor(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)
    return inpainted

# 3. Shades of Gray: Color constancy
def shades_of_gray(img: np.ndarray, power: int = 6, gamma: float = None) -> np.ndarray:
    img = img.astype(np.float32)
    if gamma is not None:
        img = img ** (1.0 / gamma)
    mean_per_channel = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1/power)
    norm = np.sqrt(np.sum(np.power(mean_per_channel, 2.0)))
    scale = norm / mean_per_channel
    img = img * scale
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# 4. Center-crop to square, then resize
def center_crop_and_resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return resized

# 5. Metadata encoding for ISIC (age, sex, site)
def encode_metadata(row: pd.Series, age_bins: List[int] = [0, 20, 40, 60, 80, 120]) -> np.ndarray:
    # Age binning (one-hot)
    age = row.get('age_approx', np.nan)
    age_bin = np.digitize([age], age_bins)[0] if not np.isnan(age) else 0
    age_onehot = np.zeros(len(age_bins)+1)
    age_onehot[age_bin] = 1
    # Sex encoding
    sex = row.get('sex', '').lower()
    sex_map = {'male': 0, 'female': 1}
    sex_onehot = np.zeros(2)
    if sex in sex_map:
        sex_onehot[sex_map[sex]] = 1
    # Site encoding (site_naevus, site_torso, etc.)
    site = row.get('anatom_site_general_challenge', '').lower()
    site_list = ['head/neck', 'upper extremity', 'lower extremity', 'torso', 'palms/soles', 'oral/genital']
    site_onehot = np.zeros(len(site_list))
    if site in site_list:
        site_onehot[site_list.index(site)] = 1
    # Concatenate all
    return np.concatenate([age_onehot, sex_onehot, site_onehot])

# 6. Compose pipeline
def preprocess_image(img_path: str, dullrazor: bool = True, color_constancy: bool = True, size: int = 224) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dullrazor:
        img = dull_razor(img)
    if color_constancy:
        img = shades_of_gray(img)
    img = center_crop_and_resize(img, size)
    return img

# Example usage for a pipeline
class PreprocessingPipeline:
    def __init__(self, dullrazor=True, color_constancy=True, size=224):
        self.dullrazor = dullrazor
        self.color_constancy = color_constancy
        self.size = size

    def process(self, img_path: str) -> np.ndarray:
        return preprocess_image(img_path, self.dullrazor, self.color_constancy, self.size)

    def process_metadata(self, row: pd.Series) -> np.ndarray:
        return encode_metadata(row)

# Example: pipeline = PreprocessingPipeline()
# processed_img = pipeline.process('path/to/image.jpg')
# encoded_meta = pipeline.process_metadata(row)
