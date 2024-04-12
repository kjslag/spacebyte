#!/usr/bin/env python3

# to run jobs on other nodes, set SUBMIT_COMMAND to a submission command
SUBMIT_COMMAND = []
# SUBMIT_COMMAND = ['echo'] # just print the commands to run

import math
import subprocess

def submit(*, flops, **kwargs):
    kwargs['iters'] = f'{flops}/flops'
    cmd = SUBMIT_COMMAND + [
        'python3', 'train.py'] + [
        f'--{k}' if v is True else f'--{k}={v}' for k, v in kwargs.items() if v is not None]
    subprocess.run(cmd)

out_dir='spacebyte'
Ld = {192:4, 256:8, 384:16, 512:24, 768:32, 1024:32, 1536:48, 2048:48} # 32:1, 64:2, 128:3,
ds = list(Ld.keys())
Ls = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]

rope = True
beta2 = 0.98
B = 64
lr = 0.5
lr = f'{lr}e-2*{B}**0.5'
for dataset in ['github', 'arxiv', 'pg19']:
    for flops in ['1e18', '1e19']:
        for tok in [None, 'gpt2', 'sp']:
            for d in ds:
                if {'1e16': 192 <= d <= 256, '1e17': 256 <= d <= 384, '1e18': 384 <= d <= 768, '1e19': 512 <= d <= 1024}[flops]:
                    for L in Ls:
                        P = 6 if 'github' not in dataset else 8
                        args = dict(dataset=dataset, flops=flops, tokenizer=tok, batch_size=B, lr=lr, beta2=beta2,
                        context_size=d, d_model=d, n_layers=L, rope=rope, out_dir=out_dir)

                        good_L = L == Ld[d]//2 or L == Ld[d]
                        if good_L and 2 <= L:
                            if tok is None:
                                pass
                                submit(**(args | dict(context_size=P*d, attention_window=d)))
                                submit(**args)
                            else:
                                pass
                                submit(**args)

                        good_L = Ld[d]//4 < L <= Ld[d]//2
                        if good_L and 2 <= L and tok is None:
                            for d_local in ds:
                                if d//2 <= d_local < d:
                                    if d_local >= {'1e17': 192, '1e18': 256, '1e19': 384}[flops]:
                                        for L_local in [L]:
                                            for P_MB in [4, 8]:
                                                mega_args = args | dict(model='MegaByte', patch_size=P_MB, context_size=P_MB*d,
                                                    d_local=d_local, n_local_layers=L_local)
                                                submit(**mega_args)

                                            for P in [P]:
                                                for patch_method in ['utf8', 'periodic']:
                                                    TG = d if P != 6 else None
                                                    TL = P*TG if P != 6 else None
                                                    wise_args = args | dict(model='SpaceByte', patch_method=patch_method,
                                                        global_context_size=TG, context_size=TL, d_local=d_local, n_local_layers=L_local,
                                                        local_attention_window=d_local)
                                                    submit(**wise_args)
