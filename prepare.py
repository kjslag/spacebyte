#!/usr/bin/env python3

import os
from tqdm import tqdm

import numpy as np
import datasets

import util

def prepare(dataset_name: str,
            name: str = None,
            max_data_bytes: str = -1, # max number of bytes from dataset to use
            tokenizer: str = None,
            test_fraction: float = 0.01, # fraction of train data to allocate to test and validation if they don't exist
            out_dir: str = None,
            filter: str = None,
):
    if isinstance(max_data_bytes, str):
        max_data_bytes = round(eval(max_data_bytes))
    if name and ',' in name:
        name = tuple(name.split(',')) # fire does this automatically

    if out_dir is None:
        out_dir = os.path.join('datasets', dataset_name
            + (f'_{name}' if name else '')
            + (f'.{filter.split(".")[-1]}' if filter else '')
        )
    os.makedirs(out_dir, exist_ok=True)
    gen = np.random.default_rng(0)

    if tokenizer is not None and tokenizer not in util.Tokenizer.tiktoken_encodings:
        tokenizer = f'{out_dir}/{tokenizer}'
    tokenizer = util.Tokenizer(tokenizer)

    def open_split(split):
        file_name = os.path.join(out_dir, split + tokenizer.file_suffix)
        assert not os.path.exists(file_name)
        return open(file_name, 'wb')

    def get_dataset(name):
        if name is not None:
            print('loaded', name)
        return datasets.load_dataset(dataset_name, name, streaming=True, trust_remote_code=True)
    if isinstance(name, tuple):
        dataset_list = [get_dataset(name0) for name0 in name]
        def merged_dataset(split):
            merge_gen = np.random.default_rng(abs(hash(split)))
            dataset_iters = [iter(d[split]) for d in dataset_list]
            while True:
                if len(dataset_iters) == 0:
                    break
                i = merge_gen.integers(len(dataset_iters))
                try:
                    yield next(dataset_iters[i])
                except StopIteration:
                    del dataset_iters[i]
        dataset = {split: merged_dataset(split) for split in dataset_list[0].keys()}
    else:
        dataset = get_dataset(name)

    print(f'found splits: {list(dataset.keys())}')
    for split, data in dataset.items():
        total_data_bytes = 0
        with open_split(split) as out_file:
            test_files = []
            if split == 'train':
                for s in 'test', 'validation':
                    if s not in dataset:
                        print(f'{s} not in dataset. Randomly allocating {test_fraction*100:.2g}% of train to {s}...')
                        test_files.append(open_split(s))

            for example in tqdm(data, split):
                if filter is not None:
                    *filter_keys, filter_value = filter.split('.')
                    ex = example
                    for key in filter_keys:
                        ex = ex[key]
                    if ex != filter_value:
                        continue
 
                key = 'code' if 'github-code' in dataset_name else \
                      'content' if 'the-stack' in dataset_name else 'text'
                text = example[key]

                if max_data_bytes > 0:
                    total_data_bytes += len(text.encode('utf-8'))
                    if total_data_bytes > max_data_bytes:
                        break

                text = tokenizer.encode(text, prepend_BOS=True, dtype=tokenizer.dtype, tensor=np.array)
                assert tokenizer.BOS not in text[1:]
                text = text.tobytes()
                write_text = True

                if len(test_files) > 0:
                    r = gen.random()
                    for f in test_files:
                        if r < test_fraction:
                            f.write(text)
                            write_text = False
                            break
                        else:
                            r -= test_fraction

                if write_text:
                    out_file.write(text)

            for f in test_files:
                f.close()

import fire
if __name__ == '__main__':
    fire.Fire(prepare)
