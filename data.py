import os

import numpy as np
import torch

import util

def dataset(name, tokenizer=None):
    if name == 'zeros':
        return ZeroesDataset(tokenizer)
    else:
        return MemmapDataset(name, tokenizer)

class MemmapDataset:
    def __init__(self, name, tokenizer=None):
        super().__init__()

        if not isinstance(tokenizer, util.Tokenizer):
            if tokenizer is not None and tokenizer not in util.Tokenizer.tiktoken_encodings:
                tokenizer = f'datasets/{name}/{tokenizer}'
            tokenizer = util.Tokenizer(tokenizer)
        self.tokenizer = tokenizer

        data_dir = os.path.join('datasets', name)
        self.data = {}
        self.bytes_per_token = None
        for file_name in os.listdir(data_dir):
            if file_name.endswith(tokenizer.file_suffix):
                split = file_name[:-len(tokenizer.file_suffix)]
                # we recreate np.memmap with every access to avoid a memory leak
                # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
                self.data[split] = lambda file_name=file_name: \
                    np.memmap(os.path.join(data_dir, file_name), dtype=tokenizer.dtype, mode='r')
                if split == 'train':
                    # NOTE: assume that the tokenized file contains the same data as the .txt file
                    self.bytes_per_token = 1 if tokenizer.name is None else (
                        os.path.getsize(os.path.join(data_dir, split + '.txt')) * np.dtype(tokenizer.dtype).itemsize /
                        os.path.getsize(os.path.join(data_dir, file_name)) )
        print(f'MemmapDataset: found {list(self.splits())} splits')
        assert self.bytes_per_token is not None

        if 'validation' in self.data:
            assert 'val' not in self.data
            self.data['val'] = self.data['validation']
            del self.data['validation']

        self.vocab_size = tokenizer.vocab_size
        self.BOS = tokenizer.BOS

    def splits(self):
        return self.data.keys()

    def iter(self, split, *, context_size, batch_size=1, seed=0, device='cpu'):
        data = self.data[split]
        data_size = len(data())
        T = context_size
        B = batch_size
        rand_gen = torch.Generator()
        rand_gen.manual_seed(seed)

        while True:
            targets = torch.zeros(B, T, dtype=torch.int64)
            tokens = torch.full((B, T), self.BOS, dtype=torch.int64)
            
            b = 0
            while b < B:
                t = torch.randint(data_size, tuple(), generator=rand_gen)
                target = data()[t:]

                # align with BOS if found in next T tokens
                BOS_index, = (target[:T] == self.BOS).nonzero()
                if len(BOS_index) > 0:
                    target = target[BOS_index[0]+1:]

                target = target[:T]
                if len(target) < T:
                    continue
                targets[b] = torch.from_numpy(target.astype(np.int64))
                b += 1

            tokens[:,1:] = targets[:, :-1]
            yield to_device(tokens, device), to_device(targets, device)

class ZeroesDataset:
    def __init__(self, tokenizer=None):
        if not isinstance(tokenizer, util.Tokenizer):
            tokenizer = util.Tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.vocab_size = tokenizer.vocab_size
        self.BOS = tokenizer.BOS
        self.bytes_per_token = 1

    def splits(self):
        return ['train', 'val', 'test']

    def iter(self, split, *, context_size, batch_size=1, seed=0, device='cpu'):
        T = context_size
        B = batch_size

        while True:
            tokens = torch.full((B, T), 0)
            targets = torch.cat([torch.full((B, 1), self.BOS), tokens[:, :-1]], 1)
            yield to_device(tokens, device), to_device(targets, device)

def to_device(x, device):
    if 'cuda' in device:
        x = x.pin_memory()
    if 'cpu' not in device:
        # non_blocking=True is bugged on mps
        # https://github.com/pytorch/pytorch/issues/83015
        x = x.to(device, non_blocking = 'mps' not in device)
    return x
