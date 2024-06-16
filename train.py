import os
from os.path import join as path_join
import json
import random
import argparse

from tqdm.auto import tqdm

import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision
import torchvision.transforms.v2 as T
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

from data import AidaDataset
from transforms import FixedAspectResize, RandomSpots
from tokenizers import LaTeXTokenizer
from models import LaTeXOCREncoder, LaTeXOCRDecoder, LaTeXOCRModel, LitLaTeXOCRModel


DATA_DIR = "data/cleaned_aida"

SEED = 101
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_SIZE = 512
BATCH_SIZE = 8
LR = 1e-3


def build_ocr_model(tokenizer: LaTeXTokenizer):
    encoder_backbone = torchvision.models.efficientnet_v2_m().features[:-1]
    encoder = LaTeXOCREncoder(
        encoder_backbone, 
        d_backbone=512, 
        d_model=128,
    )

    decoder = LaTeXOCRDecoder(
        tokenizer.vocab_size, 
        d_model=128, 
        n_heads=4, 
        ff_dim=256, 
        n_layers=3, 
        dropout=0.1,
    )

    model = LaTeXOCRModel(encoder, decoder, tokenizer)
    model = model.to(DEVICE)

    return model


if __name__ == "__main__":
    print("Loading data ...")
    train_transform = T.Compose([
        RandomSpots(spots_range=(3, 7), w_range=(5, 10), h_range=(3, 5)),
        T.ToImage(),
        FixedAspectResize(512),
        T.ToDtype(torch.float32),
    ])
    test_transform = T.Compose([
        T.ToImage(),
        FixedAspectResize(512),
        T.ToDtype(torch.float32)
    ])
    tokenizer = LaTeXTokenizer.load_from(path_join(DATA_DIR, "vocab.json"))
    train_set = AidaDataset(DATA_DIR, tokenizer, mode="train", transform=train_transform)
    test_set = AidaDataset(DATA_DIR, tokenizer, mode="test", transform=test_transform)

    def _collate_fn(batch):
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs)
        targets = pad_sequence(targets, padding_value=tokenizer.pad_token_id, batch_first=True)

        return inputs, targets

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=_collate_fn, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=_collate_fn, num_workers=4, pin_memory=True)
    
    # build model
    model = build_ocr_model(tokenizer)

    # training
    logger = MLFlowLogger(experiment_name="CV2024 Term Project", log_model=True)
    trainer = L.Trainer(min_epochs=1, max_epochs=100, check_val_every_n_epoch=1, logger=logger)
    lit_model = LitLaTeXOCRModel(model, lr=LR, weight_decay=0.01)
    trainer.fit(lit_model, train_loader, test_loader)
