from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from tokenizers import LaTeXTokenizer
from metrics import CharacterErrorRate


def _create_causal_mask(size, dtype=None, device=None):
    """Generate a causal mask with given size. Used to mask out the yet generated latex tokens."""
    return torch.triu(
        torch.full((size, size), float('-inf'), dtype=dtype, device=device), 
        diagonal=1
    )


class PositionalEncoding1d(nn.Module):
    """1D positional encoding used in the latex decoder"""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len

        _data = self.create(d_model, max_len)
        self.register_buffer("_data", _data)

    @staticmethod
    def create(d_model: int, max_len: int = 1024):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        denominator = torch.exp(-(torch.arange(0, d_model, 2).float() / d_model) * math.log(10_000.0))
        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)

        return pe

    def forward(self, x: torch.Tensor):
        _, L, E = x.size()
        assert E == self.d_model

        return x + self._data[:L]


class PositionalEncoding2d(nn.Module):
    """2D positional encoding used in the image encoder"""
    def __init__(self, d_model: int, max_h: int = 1024, max_w: int = 1024):
        super().__init__()

        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w

        _data = self.create(d_model, max_h, max_w)
        self.register_buffer("_data", _data)

    @staticmethod
    def create(d_model: int, max_h: int = 1024, max_w: int = 1024):
        pe_h = PositionalEncoding1d.create(d_model=d_model // 2, max_len=max_h)  # (max_h, d_model // 2)
        pe_h = pe_h.unsqueeze(1).expand(-1, max_w, -1).permute(2, 0, 1)  # (d_model // 2, max_h, max_w)

        pe_w = PositionalEncoding1d.create(d_model=d_model // 2, max_len=max_w)  # (max_w, d_model // 2)
        pe_w = pe_w.unsqueeze(0).expand(max_h, -1, -1).permute(2, 0, 1)  # (d_model // 2, max_h, max_w)

        pe = torch.cat([pe_h, pe_w], dim=0)  # (d_model, max_h, max_w)
        return pe

    def forward(self, x: torch.Tensor):
        _, C, H, W = x.size()
        assert C == self.d_model

        return x + self._data[:, :H, :W]


class LaTeXOCREncoder(nn.Module):
    """The handwriting latex image encoder"""
    MAX_H = 1024
    MAX_W = 1024

    def __init__(self, backbone: nn.Module, d_backbone: int, d_model: int):
        super().__init__()

        self.d_backbone = d_backbone
        self.d_model = d_model

        self.channel_proj = nn.Conv2d(1, 3, kernel_size=1)
        self.backbone = backbone
        self.bottleneck = nn.Conv2d(d_backbone, d_model, kernel_size=1)
        self.positional_encoding = PositionalEncoding2d(d_model, self.MAX_H, self.MAX_W)

    def forward(self, x: torch.Tensor):
        x = self.channel_proj(x)
        x = self.backbone(x)
        x = self.bottleneck(x)
        h = self.positional_encoding(x)

        return h


class LaTeXOCRDecoder(nn.Module):
    """The decoder to predict the latex expression."""
    MAX_SEQ_LENGTH = 1024

    def __init__(self, vocab_size, d_model, n_heads, ff_dim, n_layers, dropout):
        super().__init__()

        _decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding1d(d_model, self.MAX_SEQ_LENGTH)
        self.backbone = nn.TransformerDecoder(_decoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, feat_map: torch.Tensor, to_probs=False):
        x = self.embedding(x)  # (B, S, E)
        x = self.positional_encoding(x)  # (B, S, E)

        S = x.size(1)
        causal_mask = _create_causal_mask(S, x.dtype, device=x.device)
        x = self.backbone(x, feat_map, tgt_mask=causal_mask, tgt_is_causal=True)

        h = self.classifier(x)  # (B, S, C)
        if to_probs:
            h = F.softmax(h, dim=-1)
        return h


class LaTeXOCRModel(nn.Module):
    def __init__(self, encoder: LaTeXOCREncoder, decoder: LaTeXOCRDecoder, tokenizer: LaTeXTokenizer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    def encode(self, x):
        feat_map: torch.Tensor = self.encoder(x)
        return feat_map.flatten(start_dim=2).transpose(1, 2)

    def decode(self, x, feat_map, to_probs=False):
        return self.decoder(x, feat_map, to_probs)

    def forward(self, x, y, to_probs=False):
        feat_map = self.encode(x)
        y_hat = self.decode(y, feat_map, to_probs)

        return y_hat

    def predict(self, x: torch.Tensor):
        """Generate the latex sequences of the input images `x`.

        Args:
            x (torch.Tensor): the input image batch

        Returns:
            torch.Tensor: the completed latex token sequences
        """
        B = x.size(0)
        S = self.decoder.MAX_SEQ_LENGTH
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        feat_map = self.encode(x)

        result = torch.full(size=(B, S), fill_value=pad_id, dtype=torch.int64, device=x.device)
        result[:, 0] = bos_id

        # generate the latex expressions token-by-token
        completed = torch.full(size=(B,), fill_value=False, device=x.device)
        for idx in range(1, S):
            y = result[:, :idx]
            logits = self.decode(y, feat_map, to_probs=True)  # (B, S, C)
            output = torch.argmax(logits, dim=-1)  # (B, S)
            result[:, idx] = output[:, -1]

            # stop if all samples in the batch are completed with eos tokens.
            completed |= (result[:, idx] == eos_id)
            if torch.all(completed):
                break

        # set the tokens after the eos token as pad tokens
        for i in range(B):
            eos_pos = torch.where(result[i] == eos_id)[0]
            if len(eos_pos) == 0:
                continue
            
            eos_idx = eos_pos[0]
            result[i, eos_idx+1:] = pad_id

        return result


class LitLaTeXOCRModel(L.LightningModule):
    def __init__(self, model: LaTeXOCRModel, lr: float, weight_decay: float, milestones: List[int] = [5], gamma: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.tokenizer = self.model.tokenizer

        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.cer_metric = CharacterErrorRate(
            ignore_indices={
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits: torch.Tensor = self.model(imgs, targets[:, :-1])  # (B, S, C)
        loss = self.criterion(logits.permute(0, 2, 1), targets[:, 1:])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits: torch.Tensor = self.model(imgs, targets[:, :-1])
        loss = self.criterion(logits.permute(0, 2, 1), targets[:, 1:])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = self.model.predict(imgs)
        cer = self.cer_metric(preds, targets)
        self.log("val/cer", cer)