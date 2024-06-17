import json
from typing import Dict, Sequence

import torch


class LaTeXTokenizer:
    def __init__(self, vocab: Dict[str, int], bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>"):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.id_to_token = {v: k for k, v in vocab.items()}

    @property
    def bos_token_id(self):
        return self.vocab[self.bos_token]

    @property
    def eos_token_id(self):
        return self.vocab[self.eos_token]

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token]

    def encode(self, x, add_special_tokens=False, return_tensor=False):
        """Encode tokens `x` to token ids"""
        if isinstance(x, Sequence):
            result = [self.vocab.get(token, self.unk_token_id) for token in x]
            if add_special_tokens:
                result = [self.bos_token_id] + result + [self.eos_token_id]
        else:
            result = self.vocab.get(x, self.unk_token_id)
            if add_special_tokens:
                result = [self.bos_token_id, result, self.eos_token_id]

        return torch.tensor(result) if return_tensor else result
        
    def decode(self, x):
        """Decode token ids `x` to tokens"""
        if isinstance(x, Sequence):
            return [self.id_to_token[token_id] for token_id in x]
        else:
            return self.id_to_token[x]

    @classmethod
    def load_from(cls, vocab_file):
        """Load vocabulary from the file."""
        with open(vocab_file) as f:
            _vocab = json.load(f)

        return cls(_vocab)
