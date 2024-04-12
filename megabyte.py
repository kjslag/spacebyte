from transformer import *

@dataclass
class MegaByteConfig(TransformerConfig):
    patch_size: int = 8

    d_local: int = None
    n_local_layers: int = None

    # I don't think this is needed, but I include it to follow the MegaByte paper.
    use_padding: bool = True

    def __post_init__(self):
        c = self

        if c.context_size is None:
            c.context_size = c.patch_size * c.d_model

        super().__post_init__()

        if c.d_local is None:
            c.d_local = c.d_model//2
        if c.n_local_layers is None:
            c.n_local_layers = c.n_layers

class MegaByte(Model):
    Config = MegaByteConfig

    def __init__(self, config: MegaByteConfig):
        super().__init__()
        self.config = config
        c = config

        T = c.context_size
        K = int_div(T, c.patch_size)
        D_G = int_div(c.d_model, c.patch_size)

        assert not c.tie_embedding # not implemented
        self.global_token_embedding = nn.Embedding(c.vocab_size, D_G)
        assert c.position_encoding # else not implemented
        self.global_position_encoding = nn.Parameter(torch.randn(T, D_G))

        if c.use_padding:
            self.global_pad = nn.Parameter(torch.randn(c.d_model))
        
        global_config = c.copy(context_size=K)
        self.global_blocks = nn.ModuleList([TransformerBlock(global_config) for _ in range(c.n_layers)])

        self.global_to_local = nn.Linear(D_G, c.d_local)

        self.local_token_embedding = nn.Embedding(c.vocab_size, c.d_local)
        # Local position encoding does not appear in Fig. 2 of the MegaByte paper, but it is used according to:
        # https://openreview.net/forum?id=JTmO2V9Xpz&noteId=VhgZzXezYZ
        self.local_position_encoding = nn.Parameter(torch.randn(c.patch_size, c.d_local))
        
        if c.use_padding:
            self.local_pad = nn.Parameter(torch.randn(c.d_local))

        local_config = c.copy(d_model=c.d_local, context_size=c.patch_size, attention_window=None)
        self.local_blocks = nn.ModuleList([TransformerBlock(local_config) for _ in range(c.n_local_layers)])

        self.logits = Logits(local_config)

        super().__post_init__()

    generate = Transformer.generate
    train_log = Transformer.train_log

    def num_params(self, embedding=True):
        n = num_params(self)
        if not embedding:
            n -= num_params(self.global_token_embedding)
            n -= num_params(self.global_position_encoding)
            n -= num_params(self.local_token_embedding)
            n -= num_params(self.local_position_encoding)
        return n

    def n_mult_add(self, training=False):
        c = self.config
        P = c.patch_size
        T = c.context_size
        K = T // P
        d = c.d_local
        V = c.vocab_size

        n  = sum(module.n_mult_add(K) for module in self.global_blocks)
        n += T * num_params(self.global_to_local)
        n += K * sum(module.n_mult_add(P) for module in self.local_blocks)

        return n + T*d*V

    def forward(self, tokens, targets=None, *, cache=None, log=None):
        c = self.config
        B, T0 = tokens.shape
        P = c.patch_size
        D = c.d_model
        D_G = int_div(D, P)
        d = c.d_local

        if cache is None:
            t0 = 0
            pending_global_tokens = None
        else:
            prefix = self.module_name + '->'
            t0 = cache.get(prefix+'t0', 0)
            cache[prefix+'t0'] = t0 + T0
            pending_global_tokens = cache.get(prefix+'pending_global_tokens', None)
        if pending_global_tokens is None:
            pending_global_tokens = torch.full((B, P-1), c.BOS, device=tokens.device)

        global_tokens = torch.cat([pending_global_tokens, tokens], 1)
        Kx = global_tokens.shape[1] // P
        if cache is not None:
            cache[prefix+'pending_global_tokens'] = global_tokens[:, Kx*P:]

        if Kx > 0:
            global_tokens = global_tokens[:, :Kx*P]
            global_t = (P-1 + t0) // P

            x = self.global_token_embedding(global_tokens) # (B, Kx*P, D_G)
            x = x + self.global_position_encoding[global_t*P : (global_t+Kx)*P]
            x = x.view(B, Kx, D)

            if c.use_padding and global_t == 0:
                x = torch.cat([self.global_pad.broadcast_to(B,1,D), x[:,1:]], 1) # B, K, D

            for block in self.global_blocks:
                x = block(x, cache=cache, cache_seqlen=global_t, log=log)

            x = x.view(B*Kx, P, D_G)
            x = self.global_to_local(x).view(B, Kx*P, d)

        local_emb = self.local_token_embedding(tokens) # (B, T0, d)
        if t0==0 and T0>=P:
            t1 = (T0//P)*P
            y = local_emb[:, :t1].reshape(B*(T0//P), P, d) + self.local_position_encoding
            local_emb = local_emb[:, t1:] # (B, T0-t1, d)

            if c.use_padding:
                y = torch.cat([self.local_pad.broadcast_to(len(y), 1, d), y[:, 1:]], 1)

            y = x[:, :t1].reshape(-1, P, d) + y
            x = x[:, t1:]

            for block in self.local_blocks:
                y = block(y, log=log) # cache not needed

            y = y.view(B, t1, d)
        else:
            t1 = t0

        T1 = local_emb.shape[1]
        if cache is not None:
            if t0 > 0:
                x = torch.cat([cache[prefix+'local-in'], x], 1) if Kx>0 else cache[prefix+'local-in']
            cache[prefix+'local-in'] = x[:, T1:]
            assert (t1 + T1 + x[:, T1:].shape[1]) % P == 0
            x = x[:, :T1]

        if T1 > 0:
            assert t1//P == (t1 + T1 - 1)//P
            z = local_emb + self.local_position_encoding[t1%P : (t1+T1-1)%P + 1] # (B, T1, d)

            if c.use_padding and t1%P == 0:
                z = torch.cat([self.local_pad.broadcast_to(B, 1, d), z[:, 1:]], 1)

            z = x + z

            if t1%P == 0:
                cache[prefix+'local-cache'] = {}
            for block in self.local_blocks:
                z = block(z, cache=cache[prefix+'local-cache'], cache_seqlen=t1%P, log=log)

            y = torch.cat([y, z], 1) if t0==0 and T0>=P else z

        logits = self.logits(y, log=log).view(B, T0, c.vocab_size)

        losses = None
        if targets is not None:
            losses = {}
            losses['cross entropy'] = util.cross_entropy(logits, targets)
            losses['loss'] = losses['cross entropy']

        return logits, losses
