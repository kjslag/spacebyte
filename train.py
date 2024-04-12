#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import shutil
import time
import datetime
import math
# import itertools
import copy
import contextlib
import collections
import dataclasses
from dataclasses import dataclass
# from typing import List, Optional, Tuple, Union
from tqdm import tqdm

import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, NAdam, AdamW # needed for TrainConfig.optimizer

import util
from util import notNone
import data
from transformer import Transformer, TransformerConfig, NoModel
from spacebyte import SpaceByte, SpaceByteConfig
from megabyte import MegaByte, MegaByteConfig

@dataclass
class TrainConfig:
    # model
    model: str = 'Transformer'
    model_seed: int = 0

    # command line
    args: dict = None
    note: str = ''

    # data
    data_seed: int = 0
    dataset: str = 'pg19'
    batch_size: int = 64
    micro_batch_size: int = None

    # checkpoint
    load_checkpoint_dir: str = None
    checkpoint: bool = True
    checkpoint_model: bool = True
    min_checkpoint_interval: float = 10*60 # in seconds

    # optimization
    lr: str = '0.5e-2 * B**0.5' # can use: B, T, d
    iters: str = '20 * N / tokens' # can use: N, num embedding E, tokens, and flops
    optimizer: str = 'AdamW'
    optimizer_kwargs: str = 'dict(fused=using_cuda)'
    decay_lr: str = 'half-cos' # 'half-cos', 'cos'
    warmup_iters: str = 'iters/100'
    beta1: float = 0.9
    beta2: str = 0.98
    # note: pytorch does not implement weight_decay as in "Decoupled Weight Decay Regularization" since pytorch multiplies the weight decay by the max learning rate while the paper does not
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # system
    device: str = None # 'cpu', 'mps', 'cuda'
    compile: bool = False
    dtype: str = None # 'float32' for no autocast. 'bfloat16' or 'float16' for autocast. 'float16' will use a GradScaler
    use_deterministic: bool = False

    # logging
    out_dir: str = 'out' # set to '' to disable. auto subdir unless trailing /
    log_interval: float = 3 # in seconds

    # evaluating
    eval_interval: str = 'iters / 50'
    eval_iters: str = 'eval_interval/40 * B/mB' # number of micro-batches for mid-training evaluation
    final_eval_iters: str = 'min(iters/30, 2**14) * B/mB' # number of micro-batches for the final evaluation

    # wandb logging
    wandb_log: bool = None
    wandb_project: str = None
    wandb_run_name: str = None
    
    # debug
    checkpoint_nan: bool = False
    check_nan: bool = False

    def __post_init__(self):
        if self.device is None:
            self.device = util.default_device()
        
        if self.compile is None:
            self.compile = 'cuda' in self.device

        if self.dtype is None:
            # todo M2 supports bfloat16 https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
            self.dtype = 'bfloat16' if 'cuda' in self.device and torch.cuda.is_bf16_supported() else 'float32'

        if self.wandb_log is None:
            self.wandb_log = self.out_dir != 'out' and self.out_dir[-1] != '/'

        if self.wandb_project is None:
            self.wandb_project = self.out_dir
        if self.args is not None:
            def run_args(sep=' '):
                def simplify(v):
                    if isinstance(v, tuple):
                        return ','.join(str(x) for x in v)
                    else:
                        return v
                return sep.join(f'--{k}={simplify(v)}' for k, v in sorted(self.args.items()))
            if self.wandb_run_name is None:
                self.wandb_run_name = run_args()
            if self.out_dir and self.out_dir[-1] != '/':
                self.out_dir = os.path.join(self.out_dir, 'Train' + run_args(''))

class Train:
    def from_checkpoint(dir_name, device=None,
                        train_config_override=dict(wandb_log=False, out_dir=''),
                        model_config_override=dict()):
        if device is None:
            device = util.default_device()
        model, checkpoint = Train.model_from_checkpoint(dir_name, device, config_override=model_config_override)

        train_config = checkpoint['train_config']
        for k in list(train_config):
            if not hasattr(TrainConfig, k):
                print(f"TrainChar: WARNING! '{k}' no longer in TrainCharConfig")
                del train_config[k]
        train_config['device'] = device
        train_config |= train_config_override
        train_config = TrainConfig(**train_config)

        return Train(model, train_config, checkpoint=checkpoint)
    
    def __init__(self, model_or_config, train_config: TrainConfig, checkpoint=None, verbose=None):
        super().__init__()
        self.train_config = train_config
        c = self.train_config
        self._estimate_losses_dataset_iters = {}

        # c.out_dir might still be ''
        if c.out_dir:
            print('out_dir =', c.out_dir)
            os.makedirs(c.out_dir, exist_ok = not c.wandb_log)
            code_dir = os.path.dirname(os.path.realpath(__file__))
            for f in os.listdir(code_dir):
                if f[-3:] == '.py':
                    shutil.copy(os.path.join(code_dir, f), c.out_dir)
            self.tee = util.Tee(os.path.join(c.out_dir, 'stdout.txt'))

        if verbose is None:
            verbose = not (util.interactive_mode() and checkpoint is not None)

        if c.use_deterministic:
            torch.use_deterministic_algorithms(True)
        
        if c.check_nan:
            torch.set_anomaly_enabled(True, check_nan=True)

        self.autocast = util.autocast_context(c.dtype)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled = 'float16' == c.dtype)

        if isinstance(model_or_config, nn.Module):
            self.model = model_or_config
            model_config = self.model.config
        else:
            self.model = None
            model_config = model_or_config

        self.dataset = data.dataset(c.dataset, model_config.tokenizer)
        if model_config.vocab_size is None:
            model_config.vocab_size = self.dataset.vocab_size
        else:
            assert model_config.vocab_size == self.dataset.vocab_size
        if model_config.BOS is None:
            model_config.BOS = self.dataset.BOS
        else:
            assert model_config.BOS == self.dataset.BOS

        self.checkpoint_vars = ['iter_num', 'decay_lr_from_iter', 'lr_sum', 'best_val_loss', 'total_flops', 'total_tokens',
            'train_time', 'eval_time']
        if c.load_checkpoint_dir or checkpoint is not None:
            checkpoint = c.load_checkpoint_dir if checkpoint is None else checkpoint
            device = self.train_config.device
            if self.model is None:
                model_config_override = {k: v for k, v in c.args.items() if hasattr(model_config, k)}
                self.model, checkpoint = Train.model_from_checkpoint(checkpoint, device, config_override=model_config_override)
            for var in self.checkpoint_vars:
                if var in checkpoint:
                    setattr(self, var, checkpoint[var])
            self.iter_num += 1
            # we load the optimizer checkpoint after initializing it below
        else:
            checkpoint = None
            torch.manual_seed(c.model_seed)
            self.model = eval(c.model)(model_config)
            self.model.to(c.device)
            for var in self.checkpoint_vars:
                setattr(self, var, 0)
            self.best_val_loss = math.inf
        mc = self.model.config
        self.model.dataset_tokenizer = self.dataset.tokenizer # useful for debugging

        N = self.model.num_params()
        N_E = self.model.num_params(embedding=False)
        T = mc.context_size
        B = c.batch_size

        if c.micro_batch_size is None:
            c.micro_batch_size = c.batch_size
        assert c.batch_size % c.micro_batch_size == 0
        mB = c.micro_batch_size

        if isinstance(c.iters, str):
            flops = c.batch_size * self.model.n_flops(average=True)
            tokens = c.batch_size * T
            N = N_E
            c.iters = round(eval(c.iters))
            N = self.model.num_params()
            del flops, tokens
        iters = c.iters

        if isinstance(c.beta2, str):
            c.beta2 = eval(c.beta2)

        if isinstance(c.eval_interval, str):
            c.eval_interval = math.ceil(eval(c.eval_interval))
        
        if isinstance(c.final_eval_iters, str):
            c.final_eval_iters = math.ceil(eval(c.final_eval_iters))
            if 0 < c.final_eval_iters < 3:
                c.final_eval_iters = 3 # avoid a strange wandb error
        if isinstance(c.eval_iters, str):
            eval_interval = c.eval_interval
            c.eval_iters = math.ceil(eval(c.eval_iters))
            if 0 < c.eval_iters < 3:
                c.eval_iters = 3 # avoid a strange wandb error

        d = mc.d_model
        if isinstance(c.lr, str):
            c.lr = eval(c.lr)
        lr = c.lr

        if isinstance(c.warmup_iters, str):
            c.warmup_iters = eval(c.warmup_iters)

        self.meta = {
            'parameters': N,
            'non-embedding parameters': N_E,
            'mult-add per token': self.model.n_mult_add() / mc.context_size,
            'train FLOPs per token': self.model.n_flops(average=True) / mc.context_size,
            'train FLOPs': c.iters * B * self.model.n_flops(average=True),
            'train tokens': c.iters * B * T,
            'bytes per token': self.dataset.bytes_per_token,
        }
        if verbose:
            b = {'float32': 4, 'bfloat16': 2, 'float16': 2}[c.dtype]
            print(f'total parameters: {N/1e6:,.1f}M')
            print(f'total non-embedding parameters: {N_E/1e6:,.1f}M')
            print(f'model memory: {b*N/1e6:,.1f}MB')
            print(f'model+grad+Adam memory: {(1+1+2)*b*N/1e6:,.1f}MB')
            print(f'mB*T*d memory: {b*mB*T*d/1e6:,.1f}MB')
            print(f'mB*T*V memory: {b*mB*T*mc.vocab_size/1e6:,.1f}MB')
            print(f'train FLOPs per token = {self.meta["train FLOPs per token"] / 1e6:,.2f}M')
            print(f'mult-add per token = {self.meta["mult-add per token"] / 1e6:,.2f}M')
            print()

        param_groups = []
        for param in self.model.parameters():
            weight_decay = (param.weight_decay_mult if hasattr(param, 'weight_decay_mult') else 1) * c.weight_decay
            lr_mult = param.lr_mult if hasattr(param, 'lr_mult') else 1
            for group in param_groups:
                if math.isclose(weight_decay, group['weight_decay']) and math.isclose(lr_mult, group['lr_mult']):
                    group['params'].append(param)
                    break
            else:
                param_groups.append({'params': [param], 'weight_decay': weight_decay, 'lr_mult': lr_mult})
        if verbose and len(param_groups) > 1:
            for group in param_groups:
                group = copy.copy(group)
                group['params'] = [param.parameter_name for param in group['params']]
                print(group)
            print()

        using_cuda = 'cuda' in c.device # possibly used by eval(c.optimizer_kwargs)
        if isinstance(c.optimizer_kwargs, str):
            c.optimizer_kwargs = eval(c.optimizer_kwargs)
        self.optimizer = eval(c.optimizer)(param_groups, lr=c.lr, betas=(c.beta1, c.beta2), **c.optimizer_kwargs)
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if c.compile:
            print('compiling the model...')
            self.model = torch.compile(self.model)

        if c.wandb_log:
            wandb.init( project=c.wandb_project, name=c.wandb_run_name, save_code=True, # reinit=True,
                config=dataclasses.asdict(mc) | dataclasses.asdict(c) | self.meta )
            print()

        if verbose:
            for k, v in dataclasses.asdict(mc).items():
                print(f'{k} = {v}')
            print()
            for k, v in dataclasses.asdict(c).items():
                print(f'{k} = {v}')
            print()

    def dataset_iter(self, split, dataset=None, **kwargs):
        c = self.train_config
        mc = self.model.config
        if dataset is None:
            dataset = self.dataset
        kwargs = dict(context_size=mc.context_size, batch_size=c.micro_batch_size, seed=c.data_seed, device=c.device) | kwargs
        return dataset.iter(split, **kwargs)

    def train(self, callback=None):
        c = self.train_config
        mc = self.model.config

        print(f'tokens todo: {(c.iters-self.iter_num) * c.batch_size * mc.context_size:.4g}')
        print(f'FLOPs todo: {(c.iters-self.iter_num) * c.batch_size * self.model.n_flops(average=True):.4g}\n')

        # if c.wandb_log:
            # wandb.watch(self.model, log_freq=c.eval_interval)

        self.optimizer.zero_grad()
        data_iter = self.dataset_iter('train')
        next_tokens = next(data_iter)

        # todo skip data up to self.iter_num in case of loading from a checkpoint
        if c.data_seed == 0:
            assert self.iter_num == 0 # for now, at least just make sure we change the data seed

        train_time_t0 = time.time()
        last_log_time = train_time_t0
        last_log_iter = -1
        last_checkpoint_time = time.time()

        for iter_num in range(self.iter_num, c.iters):
            self.iter_num = iter_num
            self.model.train()
            model_flags = self.model.next_iter(iter_num / c.iters)
            if model_flags['warmup_lr_again']:
                print('Train: warmup_lr_again')
                self.decay_lr_from_iter = self.iter_num

            lr = c.lr
            if c.decay_lr:
                t = iter_num/c.iters
                if c.decay_lr[:3] == 'cos':
                    mult = eval(c.decay_lr[3:]) if len(c.decay_lr)>3 else 0.1
                    lr *= mult + (1-mult) * (math.cos(math.pi*t)+1)/2
                elif c.decay_lr == 'half-cos':
                    lr *= math.cos(0.5*math.pi*t)
                else:
                    assert False
            if c.warmup_iters > 0:
                lr = lr * min(1, (iter_num+1 - self.decay_lr_from_iter)/c.warmup_iters)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr * param_group['lr_mult']
            self.lr_sum += lr

            final_iter = iter_num == c.iters-1
            run_eval = c.final_eval_iters > 0 if final_iter else \
                c.eval_iters > 0 and c.eval_interval > 0 and iter_num % c.eval_interval == 0 and iter_num > 0

            n_micro_batches = util.int_div(c.batch_size, c.micro_batch_size)
            for b in range(n_micro_batches):
                tokens, targets = next_tokens
                model_log = None # {} if run_eval and b+1 == n_micro_batches else None
                with self.autocast:
                    _, losses = self.model(tokens, targets, log=model_log)
                loss = losses['loss']

                next_tokens = next(data_iter)

                if c.checkpoint_nan and math.isnan(loss):
                    if c.out_dir:
                        self.checkpoint(c.out_dir + '_nan')
                    raise Exception('nan')

                self.grad_scaler.scale(loss).backward()

            if c.grad_clip > 0:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # check that all parameters got a gradient, else we likely have unused parameters
            if iter_num == self.decay_lr_from_iter and model_flags['check_grads']:
                for param in self.model.parameters():
                    assert param.grad is not None, param.parameter_name

            if model_log is not None:
                model_log = model_log | self.model.train_log(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)

            flops = c.batch_size * self.model.n_flops()
            self.total_flops += flops
            self.total_tokens += n_micro_batches * tokens.numel()

            t1 = time.time()
            print_stats = run_eval or t1 - last_log_time > c.log_interval or iter_num == 0
            if print_stats:
                dt = (t1 - last_log_time) / (iter_num - last_log_iter)
                last_log_time = t1
                flops /= dt
                tps = tokens.numel() / dt
                last_log_iter = iter_num

                # print(util.chrs(tokens[0,:128].cpu()))
                # print(util.chrs(targets[0,:128].cpu()))

                # single print statement so all lines are printed at the same time
                eta = str(datetime.timedelta(seconds=round(dt*(c.iters-iter_num))))
                print(f'iter {iter_num:6} ({100*iter_num/c.iters:7.3g}%,',
                    f'{eta:15}):',
                    ', '.join(f'{name} {loss:<8.3g}' for name, loss in losses.items()) + ',',
                    f'{dt*1000:4.0f}ms, {flops/1e12:5.3g} TFLOP/s, {tps/1000:.1f}k tps,',
                    util.get_memory_stats(device=c.device) + ',',
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S') )
                assert not math.isnan(loss)
                del dt

            if callback is not None:
                callback(**locals())
            del loss, losses

            # evaluate the loss, log, and write checkpoints
            if run_eval:
                util.synchronize_device(c.device)
                util.empty_cache(c.device)
                eval_start_time = time.time()
                self.train_time += eval_start_time - train_time_t0

                ideally_val = 'val' if 'val' in self.dataset.splits() else 'train'
                if not final_iter:
                    eval_iters = c.eval_iters
                else:
                    eval_iters = c.final_eval_iters
                splits = self.dataset.splits()
                split_losses = self.estimate_losses(eval_iters, splits, continue_iter=not final_iter)

                util.synchronize_device(c.device)
                eval_dt = time.time() - eval_start_time
                self.eval_time += eval_dt

                print(f'\neval iter {iter_num}: {eval_dt*1000:,.0f}ms')
                sorted_losses = list(split_losses[ideally_val].items())
                sorted_losses.sort()
                print( f'{ideally_val} losses:',
                        ', '.join(f'{name} {loss:.4g}' for name, loss in sorted_losses if isinstance(loss, float)) )
                print()

                if c.wandb_log:
                    wandb_losses = { f'{split} {name}': loss
                               for split, losses in split_losses.items()
                               for name, loss in losses.items()
                               if 'token_XE' not in name or final_iter }
                    final_losses = {f'final {name}': loss for name, loss in split_losses.items()} if final_iter else {}
                    checkpoint_dict = { var.replace('_', ' '): getattr(self, var)
                        for var in self.checkpoint_vars if var != 'iter_num' }
                    wandb_dict = {
                            'iter': iter_num,
                            'lr': lr,
                            'FLOP/s': flops,
                            'TFLOP/s': flops / 1e12,
                            'eta': eta,
                            'trained percent': 100*iter_num/c.iters,
                            } | checkpoint_dict | wandb_losses | final_losses | (model_log or {})
                    try:
                        wandb.log(wandb_dict)
                    except Exception as e:
                        print('\nwandb.log Exception:', e)
                        print('wandb_dict = ', wandb_dict)
                        print()

                loss = split_losses[ideally_val]['loss']
                new_best = loss < self.best_val_loss
                if new_best:
                    self.best_val_loss = loss
                
                save_checkpoint = c.out_dir and c.checkpoint and ( final_iter or
                    time.time() > last_checkpoint_time + c.min_checkpoint_interval )
                if save_checkpoint:
                    best_ckpt = os.path.join(c.out_dir, 'ckpt_best_loss.pt')
                    checkpoint_dict = {'losses': split_losses}
                    if not new_best:
                        ckpt = os.path.join(c.out_dir, 'ckpt.pt')
                        if os.path.exists(ckpt):
                            shutil.move(ckpt, best_ckpt)
                    self.checkpoint(c.out_dir, checkpoint_dict)
                    last_checkpoint_time = time.time()
                    
                    if (new_best or final_iter) and os.path.exists(best_ckpt):
                        os.remove(best_ckpt)

                util.empty_cache(c.device)
                train_time_t0 = time.time()
                last_log_time = time.time()

            if c.out_dir and (Path(c.out_dir) / 'STOP').exists():
                assert False, 'STOP'

        if c.out_dir:
            with open(os.path.join(c.out_dir, 'FINISHED_TRAINING'), 'a'):
                pass

    def checkpoint(self, dir_name, checkpoint_dict):
        t0 = time.time()

        checkpoint_dict = dict(
            model = self.model.__class__.__name__,
            model_config = dataclasses.asdict(self.model.config),
            Train = self.__class__.__name__,
            train_config = dataclasses.asdict(self.train_config)
        ) | checkpoint_dict | self.meta
        for var in self.checkpoint_vars:
            checkpoint_dict[var] = getattr(self, var)

        print(f'saving checkpoint to {dir_name}')
        os.makedirs(dir_name, exist_ok=True)

        torch.save(checkpoint_dict, os.path.join(dir_name, 'ckpt_small.pt'))

        if self.train_config.checkpoint_model:
            checkpoint_dict |= dict(
                state_dict = self.model.state_dict(),
                optimizer = self.optimizer.state_dict() )
            torch.save(checkpoint_dict, os.path.join(dir_name, 'ckpt.pt'))

            print(f'checkpoint saved in {time.time()-t0:.1f} seconds')
            print()

    def model_from_checkpoint(checkpoint, device=None, config_override={}):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(os.path.join(checkpoint, 'ckpt.pt'), map_location=device)

        Model = eval(checkpoint['model'])
        config = checkpoint['model_config']
        for k in list(config):
            if not hasattr(Model.Config, k):
                print(f"Train: WARNING! '{k}' no longer in {Model.Config.__name__}")
                del config[k]
        config = config | config_override
        model_config = Model.Config(**config)

        model = Model(model_config)
        if device:
            model.to(device)

        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        return model, checkpoint

    def config_from_args(**kwargs):
        Model = eval(kwargs.get('model', 'Transformer'))
        ModelConfig = Model.Config
        model_config, train_config = util.make_dataclasses([ModelConfig, TrainConfig], **kwargs, args=kwargs)
        return model_config, train_config

    def from_args(**kwargs):
        model_config, train_config = Train.config_from_args(**kwargs)
        return Train(model_config, train_config)

    def estimate_losses(self, eval_iters, splits=None, continue_iter=False):
        # implement continue_iter
        def dataset_iter(*args, **kwargs):
            if not continue_iter:
                for x in self.dataset_iter(*args, **kwargs):
                    yield x
            else:
                # cycle through all the data
                key_kwargs = copy.copy(kwargs)
                if 'dataset' in key_kwargs:
                    key_kwargs['dataset'] = ''
                key = (*args, *sorted(key_kwargs.items()))
                iters = self._estimate_losses_dataset_iters
                while True:
                    if key not in iters:
                        iters[key] = self.dataset_iter(*args, **kwargs)
                    for x in iters[key]:
                        yield x
                    del iters[key]

        def _estimate_loss(dataset_iter):
            return estimate_loss(dataset_iter, eval_iters, self.model,
                bytes_per_token=self.dataset.bytes_per_token, autocast=self.autocast)

        if splits is None:
            splits = self.dataset.splits()
        losses = {split: _estimate_loss(dataset_iter(split)) for split in splits}

        return losses

def estimate_loss(dataset_iter, eval_iters, model, bytes_per_token=None, autocast=contextlib.nullcontext()):
    model.eval()
    with torch.inference_mode():
        all_losses = collections.defaultdict(util.MeanError)
        for _ in range(eval_iters):
            tokens, targets = next(dataset_iter)
            with autocast:
                logits, losses = model(tokens, targets)

            losses['token_XE'] = util.cross_entropy(logits, targets, reduction='batch', ignore_index=-1)

            losses = util.tensor_items(losses, dtype=torch.float64)
            for name, loss in losses.items():
                all_losses[name].add(loss)

        for name, losses in list(all_losses.items()):
            all_losses[name] = losses.mean()
            all_losses[name + ' stat'] = losses.error()

        if bytes_per_token is not None:
            BPB_mult = 1 / (bytes_per_token * math.log(2))
            if 'cross entropy' in all_losses:
                all_losses['bits per byte'] = BPB_mult * all_losses['cross entropy']
                all_losses['bits per byte stat'] = BPB_mult * all_losses['cross entropy stat']

        all_losses = util.tensor_items(all_losses)
        return all_losses

import sample

def train(**kwargs):
    class LastLine:
        def __init__(self):
            self.last = ''

        def write(self, data):
            if len(self.last) and self.last[-1] == '\n':
                self.last = ''
            self.last += data

        def flush(self):
            pass

    # benchmark inference
    if 'benchmark_generate' in kwargs:
        del kwargs['benchmark_generate']

        print(f'initializaing model...')
        sys.stdout = LastLine()
        trainer = Train.from_args(**kwargs)
        sys.stdout = sys.__stdout__
        tc = trainer.train_config

        B = int(kwargs.get('batch_size', 1))
        while True:
            print(f'generating batch_size={B}')
            log = sample.sample(
                model = trainer.model,
                num_samples = 2,
                batch_size = B,
                device = tc.device,
                dtype = tc.dtype,
                compile = False,
                quiet = True)
            t = log['times'][-1]
            ys = log['generations'][-1]
            print(f'generation took {t:7.3f}s, {ys[:, 1:].numel()/t/1000:6.2f} ktps,',
                util.get_memory_stats(device=tc.device))
            B *= 2

    # benchmark different batch sizes
    if kwargs.get('batch_size', '') == 'test':
        B = 1
        sys.stdout = LastLine()
        class NextBatchSize(Exception):
            pass

        while True:
            kwargs['batch_size'] = B
            sys.__stdout__.write(f'batch size = {B}:\n')
            trainer = Train.from_args(**kwargs)

            done = False
            def callback(**kwargs):
                nonlocal done
                if kwargs['print_stats']:
                    if done:
                        raise NextBatchSize
                    done = True

            try:
                trainer.train(callback=callback)
            except NextBatchSize:
                sys.__stdout__.write(sys.stdout.last)
                B *= 2

    mB = None
    out_dir = None
    while True:
        try:
            t = time.time()

            model_config, train_config = Train.config_from_args(**kwargs)
            if out_dir is None:
                out_dir = train_config.out_dir
            else:
                train_config.out_dir = out_dir
            trainer = Train(model_config, train_config)

            if mB is None:
                mB = trainer.train_config.micro_batch_size

            try:
                trainer.train()
            except BaseException as e:
                i = 1
                while os.path.exists(fail_dir := f'{out_dir}_FAILED{i}'):
                    i += 1
                shutil.move(out_dir, fail_dir)
                print(f'\nout_dir moved to {fail_dir}\n')

                trainer.tee.__del__()
                raise e

            # if trainer.train_config.wandb_log:
                # wandb.finish()
            return
        except torch.cuda.OutOfMemoryError as e:
            if mB == 1:
                print()
                print(f'micro_batch_size = {mB} can not be decreased')
                raise e

            # dt = time.time() - t
            # if dt > 30*60:
            #     print()
            #     print(f'OOM after running for {round(dt/60)} minutes')
            #     raise e

            print()
            print(e)
            print(f'Retrying with micro_batch_size = {mB} -> {mB//2} ...')
            print()
            mB //= 2
            kwargs['micro_batch_size'] = mB
            kwargs['note'] = f"{kwargs.get('note','')}_mB{mB}"
            torch.cuda.empty_cache()
            # if trainer.train_config.wandb_log:
                # wandb.finish()

import fire
if __name__ == '__main__':
    fire.Fire(train)
