import copy
import math
from dataclasses import dataclass
# from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint

try:
    import flash_attn
except:
    flash_attn = None

from rope import *

import util
from util import log_forward, num_params, notNone, like, int_div

@dataclass
class TransformerConfig:
    vocab_size: int = None
    BOS: int = None

    tokenizer: str = None
    d_model: int = 256
    context_size: int = None
    d_key: int = 64
    d_ff_mult: int = 4
    n_layers: int = None
    tie_embedding: bool = None

    # "Small-scale proxies for large-scale Transformer training instabilities"
    qk_layer_norm: bool = True

    # 'muTransfer' = "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
    init: str = 'simple' # 'standard', 'muTransfer'

    nonlinear: str = 'ReLU' # ReLU, GELU
    rope: bool = False
    position_encoding: bool = True
    attention_groups: int = None # grouped-query attention
    attention_window: int = None
    sparse_attention: bool = None # requires attention_window
    layer_norm: str = 'LayerNorm' # 'RMSNorm'

    def padded_vocab_size(self):
        d = 64
        return util.ceil(self.vocab_size, d)

    def __post_init__(self):
        d = self.d_model

        if self.context_size is None:
            self.context_size = d

        if self.n_layers is None:
            # n_layers ~ (log(d) - 5.039)/0.0555) from Eq 11 in "The Depth-to-Width Interplay in Self-Attention"
            self.n_layers = max(3, round(12.49*math.log2(d/154.3)))

        if self.tie_embedding is None:
            self.tie_embedding = self.tokenizer is not None

        assert self.init in ('standard', 'muTransfer', 'simple')

    def copy(self, **kwargs):
        c = copy.copy(self)
        for k, v in kwargs.items():
            setattr(c, k, v)
        return c

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def __post_init__(self):
        for name, module in self.named_modules():
            module.module_name = name
        for name, param in self.named_parameters():
            param.parameter_name = name

    def n_flops(self, training=True, average=False):
        return 2*(1 + 2*training)*self.n_mult_add()

    def num_params(self, embedding=True):
        return util.num_params(self)

    def next_iter(self, train_fraction):
        return dict(warmup_lr_again=False, check_grads=True)

    def train_log(self, optimizer_state):
        return {}

class NoModel(Model):
    Config = TransformerConfig

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.param = nn.Parameter(torch.tensor(1.))
        super().__post_init__()

    def n_mult_add(self):
        return 1
    
    def forward(self, tokens, targets=None, *, log=None):
        B, T = tokens.shape
        logits = torch.zeros(B, T, self.config.vocab_size, device=tokens.device)
        losses = {'loss': self.param**2}
        return logits, losses

class Transformer(Model):
    Config = TransformerConfig

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        c = self.config

        self.token_embedding = nn.Embedding(
            c.padded_vocab_size() if c.tie_embedding else c.vocab_size, c.d_model)
        if c.position_encoding:
            self.position_encoding = nn.Parameter(torch.randn(c.context_size, c.d_model))

        self.blocks = nn.ModuleList([TransformerBlock(
            c.copy(sparse_attention=(l%2) and c.sparse_attention)
            ) for l in range(c.n_layers)])

        self.logits = Logits(c, self.token_embedding if c.tie_embedding else None)

        super().__post_init__()

    def num_params(self, embedding=True):
        c = self.config
        n = num_params(self)
        if not embedding:
            if not c.tie_embedding:
                n -= num_params(self.token_embedding)
            if c.position_encoding:
                n -= num_params(self.position_encoding)
        return n

    def n_mult_add(self):
        c = self.config
        T = c.context_size
        d = c.d_model
        V = c.vocab_size
        # simple case: L*T*(4*d*d + 2*d*W + 2*d*d_ff)
        return sum(block.n_mult_add(T) for block in self.blocks) + T*d*V

    def forward(self, tokens, targets=None, *, cache=None, log=None, ignore_index=None):
        assert targets is None or cache is None
        _, Tx = tokens.shape

        x = self.token_embedding(tokens)

        if cache is None:
            t = 0
        else:
            prefix = self.module_name + '->'
            t = cache.get(prefix+'t', 0)
            cache[prefix+'t'] = t + Tx

        if self.config.position_encoding:
            x = x + self.position_encoding[t:t+Tx]

        for block in self.blocks:
            x = block(x, cache=cache, cache_seqlen=t, log=log)

        logits = self.logits(x, log=log)

        losses = None
        if targets is not None:
            losses = {}
            losses['cross entropy'] = util.cross_entropy(logits, targets, ignore_index=ignore_index)
            losses['loss'] = losses['cross entropy']

        return logits, losses

    def generate(self, tokens, *, max_tokens=None, temperature=1.0, top_k=None, input_lengths=None,
                 logits=False, use_cache=True, check_logits_func=None):
        if max_tokens is None:
            max_tokens = self.config.context_size + 1
        else:
            assert max_tokens <= self.config.context_size + 1
        B, Tx = tokens.shape
        if input_lengths is None:
            input_lengths = torch.full((B,), Tx, device=tokens.device)
        V = self.config.vocab_size
        if max_tokens > Tx:
            tokens = torch.cat([tokens, torch.zeros((B, max_tokens-Tx), **like(tokens))], 1)
        else:
            tokens = tokens[:, :max_tokens].clone()

        save_logits = logits or check_logits_func is not None
        if save_logits:
            generated_logits = torch.full((B, max_tokens - 1, V), float('nan'), device=tokens.device)

        t = 0
        cache = {}
        with torch.no_grad():
            while True:
                current_length = t + 1 if t > 0 else input_lengths.min()
                if current_length >= max_tokens:
                    break

                if use_cache:
                    next_logits, _ = self(tokens[:, t:current_length], cache=cache)
                else:
                    next_logits, _ = self(tokens[:, :current_length])
                    if check_logits_func is not None and t > 0:
                        check_logits_func(next_logits[:, :t] - generated_logits[:, :t])
                    next_logits = next_logits[:, t:current_length]

                if save_logits:
                    generated_logits[:, t:current_length] = next_logits
                t = current_length
                next_logits = next_logits[:, -1] / temperature # (B, V)
                if top_k is not None and top_k < V:
                    v, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < v[:, -1:]] = -math.inf

                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)[:,0] # (B, 1)
                next_token = next_token.where(current_length >= input_lengths, tokens[:, t])
                tokens[:, t] = next_token
        del cache

        return tokens, generated_logits if logits else None

class Logits(nn.Module):
    def __init__(self, config, tied_token_embedding=None):
        super().__init__()
        self.vocab_size = config.vocab_size

        self.layer_norm = eval(config.layer_norm)(config.d_model, config)
        self.logits_linear = tied_embedding_linear(tied_token_embedding, config) \
            if tied_token_embedding is not None else \
            Linear(config.d_model, config.padded_vocab_size(), config, output=True)

    def forward(self, x, *, log):
        x = self.layer_norm(x, log=log)
        x = self.logits_linear(x, log=log)
        x = x[:, :, :self.vocab_size] # throw away padding
        return x

def tied_embedding_linear(embedding, config):
    if config.init == 'standard':
        mult = 1
    elif config.init == 'muTransfer':
        mult = 1 / config.d_model
    elif config.init == 'simple':
        mult = config.d_model ** -0.5 # PaLM does this: "Because the input and output embedding layers are shared, we scale the pre-softmax output logits by 1/√n, where n is the embedding size."
    else:
        assert False
        
    return lambda x, *, log=None: F.linear(mult * x, embedding.weight)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = SelfAttention(config)
        self.feedforward = FeedForward(config)

    def n_mult_add(self, T):
        return self.attention.n_mult_add(T) + self.feedforward.n_mult_add(T)

    def forward(self, x, *, log, **kwargs):
        x = x + self.attention(x, log=log, **kwargs)
        x = x + self.feedforward(x, log=log)
        return x

class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        d = config.d_model
        self.d_model = d
        self.d_key = config.d_key
        self.init = config.init
        self.attention_window = config.attention_window
        self.sparse_attention = config.sparse_attention
        self.context_size = config.context_size
        assert self.attention_window is None or self.attention_window > 0
        n_head = int_div(d, self.d_key)

        self.layer_norm = eval(config.layer_norm)(d, config)
        self.attention_groups = config.attention_groups
        g = config.attention_groups
        if g is None:
            self.QKV_linear = Linear(d, 3*d, config)
        else:
            assert n_head % g == 0
            D_k = g * self.d_key
            self.Q_linear = Linear(d, d, config)
            self.KV_linear = Linear(d, 2*D_k, config)
        self.linear = Linear(d, d, config)

        self.qk_layer_norm = config.qk_layer_norm
        if self.qk_layer_norm:
            self.Q_layer_norm = eval(config.layer_norm)(self.d_key, config)
            self.K_layer_norm = eval(config.layer_norm)(self.d_key, config)

        self.rope = config.rope
        if config.rope:
            freqs_cis = precompute_freqs_cis(self.d_key, config.context_size)
            self.register_buffer('freqs_cis', freqs_cis, persistent=False)

    def n_mult_add(self, T):
        W = self.attention_window
        if W is None:
            W = T
        d = self.d_model
        return T*(4*d*d + 2*d*W)

    def forward(self, x, *, log, cache=None, cache_seqlen=None):
        if self.sparse_attention:
            B, Tx, d = x.shape
            W = self.attention_window
            x = x.view(B, Tx//W, W, d).transpose(1, 2).reshape(B*W, Tx//W, d)

        B, Tx, d = x.shape
        d_k = self.d_key
        n_h = d // d_k

        x = self.layer_norm(x, log=log)

        g = self.attention_groups
        if g is None:
            Q, K, V = self.QKV_linear(x, log=log).view(B, Tx, 3*n_h, d_k).chunk(3, dim=2)
        else:
            Q = self.Q_linear(x, log=log).view(B, Tx, n_h, d_k)
            K, V = self.KV_linear(x, log=log).view(B, Tx, 2*g, d_k).chunk(2, dim=2)

        if self.qk_layer_norm:
            Q = self.Q_layer_norm(Q, log=log)
            K = self.K_layer_norm(K, log=log)
            Q = Q.to(V.dtype)
            K = K.to(V.dtype)

        if cache is None:
            t = 0
            # ignore cache_seqlens
        else:
            prefix = self.module_name + '->'
            assert cache_seqlen is not None
            t = cache_seqlen

        if self.rope:
            Q, K = apply_rotary_emb(Q, K, self.freqs_cis[t:t+Tx])

        if self.init == 'muTransfer' and not self.qk_layer_norm:
            Q = Q * d_k**-0.5

        if cache is not None:
            if prefix+'KV' not in cache:
                cache[prefix+'KV'] = torch.stack([K, V])
            else:
                cache_KV = cache[prefix+'KV']
                cache_T = cache_KV.shape[1+1]
                if t+Tx >= cache_T:
                    padding_shape = list(cache_KV.shape)
                    padding_shape[1+1] = min(2**round(math.log2(t+Tx) + 1), self.context_size) - cache_T
                    cache_KV = torch.cat([
                        cache_KV,
                        torch.zeros(*padding_shape, **like(cache_KV))
                    ], 1+1)
                    cache[prefix+'KV'] = cache_KV

                cache_KV[0, :, t:t+Tx] = K
                cache_KV[1, :, t:t+Tx] = V
                K, V = cache_KV[:, :, :t+Tx]

        attention_window = None if self.sparse_attention else self.attention_window
        if flash_attn is not None and Q.device.type == 'cuda':
            window_size = (-1,-1) if attention_window is None else (attention_window-1, 0)
            x = flash_attn.flash_attn_func(Q, K, V, causal=True, window_size=window_size) # (B, T, n_h, d_k)
        else:
            assert Q.device.type != 'cuda' or Tx > 1 # else significant performance loss
            Tk = K.shape[1]

            mask = None
            if cache is not None or attention_window is not None:
                q_ts = torch.arange(t, t+Tx, **like(Q))[:,None]
                k_ts = torch.arange(Tk, **like(Q))
                mask = q_ts >= k_ts
                if attention_window is not None:
                    mask &= q_ts - k_ts <= attention_window

            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            if g is not None:
                Q = Q.view(B, g, n_h//g, Tx, d_k)
                K = K.view(B, g, 1, Tk, d_k)
                V = V.view(B, g, 1, Tk, d_k)
                if mask is not None:
                    mask = mask[None]

            x = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=mask is None)
            x = x.view(B, n_h, Tx, d_k).transpose(1, 2).contiguous()

        x = x.view(B, Tx, d)
        x = self.linear(x, log=log)

        if self.sparse_attention:
            BW, Tx_W, d = x.shape
            B = BW // W
            Tx = Tx_W * W
            W = self.attention_window
            x = x.view(B, W, Tx//W, d).transpose(1, 2).reshape(B, Tx, d)

        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model

        ReLU = nn.ReLU()
        GELU = nn.GELU(approximate='tanh')
        SiLU = nn.SiLU()
        self.nonlinear = eval(config.nonlinear)

        d_ff = config.d_ff_mult * d
        self.layer_norm = eval(config.layer_norm)(d, config)
        self.linear_1 = Linear(d, d_ff, config)
        self.linear_2 = Linear(d_ff, d, config)

    def n_mult_add(self, T):
        return T*(num_params(self.linear_1) + num_params(self.linear_2))

    def forward(self, x, *, log):
        x = self.layer_norm(x, log=log)
        x = self.linear_1(x, log=log)
        x = self.nonlinear(x)
        x = self.linear_2(x, log=log)
        return x

class Linear(nn.Module):
    '''Linear with normal-distributed initialization and no bias'''
    def __init__(self, d_in, d_out, config, output=False):
        super().__init__()

        self.mult = 1
        if config.init == 'standard':
            std = d_in ** -0.5
            lr_mult = 1
        elif config.init == 'muTransfer' and not output:
            # [1] "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
            # # hidden weights column of Table 3 and 8 of [1]:
            std = d_in ** -0.5
            lr_mult = 1 / d_in

            # hidden weights column of Table 8 of [1] modifed to have std = 1:
            # std = 1
            # self.mult = d_in ** -0.5
            # lr_mult = d_in ** -0.5
        elif config.init == 'muTransfer' and output:
            # output weights column of Table 3 of [1]:
            std = 1 / d_in
            lr_mult = 1 / d_in

            # output weights column of Table 8 of [1]:
            # std = 1
            # self.mult = 1 / d_in
            # lr_mult = 1
        elif config.init == 'simple':
            std = d_in ** -0.5
            lr_mult = d_in ** -0.5

            # std = 1
            # self.mult = d_in ** -0.5
            # lr_mult = 1
        else:
            assert False
        
        self.weight = nn.Parameter(torch.normal( 0, std, (d_out, d_in) ))

        if lr_mult != 1:
            self.weight.lr_mult = lr_mult

    @log_forward
    def forward(self, x, *, log):
        if self.mult != 1:
            x = self.mult * x

        return F.linear(x, self.weight)

def num_linear_params(module):
    return sum(num_params(submod) for submod in module.modules() if isinstance(submod, Linear))

class LayerNorm(nn.Module):
    def __init__(self, d, config):
        super().__init__()
        self.d = d
        self.scale = nn.Parameter(torch.ones(d))

    @log_forward
    def forward(self, x, *, log):
        return F.layer_norm(x, (self.d,), self.scale, bias=None)

class RMSNorm(nn.Module):
    def __init__(self, d, config):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))

    @log_forward
    def forward(self, x, *, log):
        eps = 1e-5
        return x * self.scale / (x.var(-1, keepdim=True, correction=0) + eps).sqrt()
