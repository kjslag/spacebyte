#!/usr/bin/env python3

import time

import torch

import util
import train

def sample(
    model: str,
    start: str = '', # can also specify a file by "FILE:prompt.txt"
    max_tokens: int = None,
    num_samples: int = 1,
    temperature: float = 1.0,
    top_k: int = None,

    seed: int = 1,
    batch_size: int = 10,
    device: str = None,
    dtype: str = None,
    compile: bool = False,

    quiet: bool = False,
    check_logits: bool = False,
    ):

    print_ = print if not quiet else (lambda *args, **kwargs: None)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if dtype is None:
        # todo M2 supposedly support bfloat16 https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
        dtype = 'bfloat16' if 'cuda' in device and torch.cuda.is_bf16_supported() else 'float32'

    if isinstance(model, str):
        model, _ = train.Train.model_from_checkpoint(model, device=device)

    if compile is None:
        compile = 'cuda' in device
    if compile:
        model = torch.compile(model)

    tokenizer = util.Tokenizer(model.config.tokenizer)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    print_(start)
    print_()
    start_tokens = tokenizer.encode(start, prepend_BOS=True, device=device)
    start_tokens = start_tokens.broadcast_to(batch_size, *start_tokens.shape)

    def check_logits_func(delta_logits):
        eps = torch.finfo(getattr(torch, dtype)).eps
        print_(f'logits error: {util.mean2(delta_logits):7.2g} '
            f'(max={delta_logits.abs().max():7.2g}, eps={eps:.2g})')
        assert util.mean2(delta_logits) < 100*eps

    torch.manual_seed(seed)
    model.eval()
    if quiet:
        log = dict(times=[], generations=[])
    with torch.inference_mode():
        with util.autocast_context(dtype):
            for _ in range(num_samples):
                t0 = time.time()
                tokens, logits = model.generate(start_tokens, max_tokens=max_tokens, temperature=temperature,
                    top_k=top_k, logits=check_logits, check_logits_func=check_logits_func if check_logits else None)
                t0 = time.time() - t0
                if check_logits:
                    T = model.config.context_size
                    forward_logits, _ = model(tokens[:, :-1][:, :T], tokens[:, 1:1+T])
                    check_logits_func(forward_logits - logits[:, :T])
                if quiet:
                    log['times'].append(t0)
                    log['generations'].append(tokens)
                print_(f'generation took {t0:.3f}s, {(tokens[:, 1:].numel() - start_tokens[:, 1:].numel())/t0:.1f} tps')
                for generation in tokens:
                    print_('---------------')
                    print_(tokenizer.decode(generation[1:]))

    if quiet:
        return log

import fire
if __name__ == '__main__':
    fire.Fire(sample)
