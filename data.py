import os
import json
from typing import Literal, Callable, Optional
from os.path import join as path_join

import numpy as np
import cv2
from tqdm.auto import tqdm

from torch.utils.data import Dataset

from tokenizers import LaTeXTokenizer


class AidaDataset(Dataset):
    """The Aida dataset including handwriting math expression images."""
    def __init__(self, 
                 data_dir: str, 
                 tokenizer: LaTeXTokenizer,
                 mode: Literal["all", "train", "test"] = "all", 
                 transform: Optional[Callable] = None):
        """init the aida dataset to have image inputs and latex targets.

        Args:
            data_dir (str): the directory to load from
            tokenizer (LaTeXTokenizer): the tokenizer to encode the latex sequence
            mode (Literal["all", "train", "test"], optional): which samples to be included. Defaults to "all".
            transform (Optional[Callable], optional): the transforms to be applied on each images sample. Defaults to None.
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform

        with open(path_join(data_dir, "train_test_split.json")) as f:
            data_split_conf = json.load(f)

        if mode != "all":
            sample_ids = data_split_conf[f"{mode}_ids"]

        self._data = []
        pbar = tqdm(sample_ids)
        for sample_id in pbar:
            pbar.set_postfix_str(sample_id)
            fp = path_join(data_dir, sample_id)

            image = cv2.imread(path_join(fp, "masked_image.png"), cv2.IMREAD_GRAYSCALE)
            with open(path_join(fp, "metadata.json")) as f:
                metadata = json.load(f)
            latex = metadata['image_data']['full_latex_chars']
            target = tokenizer.encode(latex, add_special_tokens=True, return_tensor=True)

            self._data.append((image, target))
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int):
        image, target = self._data[index]
        if self.transform:
            image = self.transform(image)

        return image, target
