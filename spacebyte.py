from transformer import *

@dataclass
class SpaceByteConfig(TransformerConfig):
    patch_method: str = 'utf8' # 'utf8', 'periodic'

    # inherited from TransformerConfig:
    # d_model = global model dimension
    # context_size = context size in bytes
    # n_layers = number of layers for the global model

    # for efficient training, global_context_size should be roughly equal to context_size/P where P is the average patch size
    global_context_size: int = None

    d_local: int = None
    n_initial_layers: int = None # number of layers for the first local model
    n_local_layers: int = None # total number of layers for both local models
    local_attention_window: int = None

    print_patches: float = 0 # fraction of time to print the patches

    def __post_init__(self):
        c = self

        if self.n_layers is None:
            # half of the Transformer default
            self.n_layers = round(max(4, 12.49*math.log2(c.d_model/154.3)) / 2)

        default_patch_size: int = 6
        if c.global_context_size is None:
            c.global_context_size = c.context_size//default_patch_size if c.context_size else self.d_model
        if c.context_size is None:
            c.context_size = default_patch_size * c.global_context_size
        if c.patch_method == 'periodic':
            assert c.context_size % c.global_context_size == 0

        super().__post_init__()

        assert c.tokenizer is None

        if c.d_local is None:
            c.d_local = c.d_model // 2
        if c.n_local_layers is None:
            c.n_local_layers = c.n_layers
        if c.n_initial_layers is None:
            c.n_initial_layers = c.n_local_layers // 2
        if c.local_attention_window is None:
            c.local_attention_window = c.d_local

class SpaceByte(Model):
    Config = SpaceByteConfig

    def __init__(self, config: SpaceByteConfig):
        super().__init__()
        self.config = config
        c = self.config

        self.token_embedding = nn.Embedding(
            c.padded_vocab_size() if c.tie_embedding else c.vocab_size, c.d_local)
        assert c.position_encoding # else not implemented
        self.local_position_encoding = nn.Parameter(torch.randn(c.context_size, c.d_local))

        local_config = c.copy(d_model=c.d_local, attention_window=c.local_attention_window)
        self.initial_blocks = nn.ModuleList([TransformerBlock(local_config) for _ in range(c.n_initial_layers)])

        self.global_position_encoding = nn.Parameter(torch.randn(c.global_context_size, c.d_model))

        global_config = c.copy(context_size=c.global_context_size)
        self.global_blocks = nn.ModuleList([TransformerBlock(global_config) for l in range(c.n_layers)])

        self.final_blocks = nn.ModuleList([
            TransformerBlock(local_config) for _ in range(c.n_initial_layers, c.n_local_layers) ])
        self.logits = Logits(local_config, self.token_embedding if c.tie_embedding else None)

        super().__post_init__()

    def num_params(self, embedding=True):
        n = num_params(self)
        if not embedding:
            if not self.config.tie_embedding:
                n -= num_params(self.token_embedding)
            n -= num_params(self.local_position_encoding) + num_params(self.global_position_encoding)
        return n

    def generate(self, tokens, *, max_tokens=None, temperature=1.0, top_k=None, input_lengths=None, logits=False,
            check_logits_func=None):
        return Transformer.generate(self, tokens, max_tokens=max_tokens, temperature=temperature, top_k=top_k,
            input_lengths=input_lengths, logits=logits, check_logits_func=check_logits_func, use_cache=False)

    def n_mult_add(self, training=False):
        c = self.config
        TL = c.context_size
        TG = c.global_context_size
        d = c.d_local
        V = c.vocab_size

        n  = sum(module.n_mult_add(TL) for module in self.initial_blocks + self.final_blocks)
        n += sum(module.n_mult_add(TG) for module in self.global_blocks)

        return n + TL*d*V

    def forward(self, tokens, targets=None, *, log=None):
        c = self.config

        x = self.token_embedding(tokens)
        B, Tx, d = x.shape

        assert Tx <= c.context_size
        x = x + self.local_position_encoding[:Tx]

        for block in self.initial_blocks:
            x = block(x, log=log)

        D = c.d_model
        T = c.context_size
        TG = c.global_context_size
        P = T // TG
        if c.patch_method != 'periodic':
            global_T = torch.full((B,), -1)
            max_global_T = min(TG, Tx)
            global_ts = torch.full((B, max_global_T), Tx-1, device=tokens.device)
            stop_gen = []
            def set_global_ts(use_global):
                for b, use_global0 in enumerate(use_global):
                    global_ts0, = use_global0.nonzero(as_tuple=True)
                    if len(global_ts0) > TG:
                        if targets is not None:
                            targets[b, global_ts0[TG]:] = -1
                        else:
                            stop_gen.append((b, global_ts0[TG]))
                    global_ts0 = global_ts0[:TG]
                    global_T[b] = len(global_ts0)
                    assert global_T[b] <= max_global_T
                    global_ts[b, :global_T[b]] = global_ts0

            if c.patch_method == 'utf8':
                # https://en.wikipedia.org/wiki/UTF-8#Encoding
                # https://en.wikipedia.org/wiki/UTF-8#Codepage_layout
                use_global = (
                    (tokens < ord('0')) |
                    ((ord('9') < tokens) & (tokens < ord('A'))) | 
                    ((ord('Z') < tokens) & (tokens < ord('a'))) |
                    ((ord('z') < tokens) & (tokens < 0b1000_0000)) |
                    (0b1100_0000 <= tokens)
                )
            else:
                assert False

            use_global[:, 1:] &= use_global[:, :-1].bitwise_not()

            if c.patch_method == 'utf8':
                use_global |= tokens == c.BOS
            set_global_ts(use_global)

            if log is not None:
                log['global_T'] = global_T
                log['global_ts'] = global_ts

            y = x.gather(1, global_ts[:, :, None].expand(B, max_global_T, d))
        else:
            y = x[:, ::P]
            max_global_T = y.shape[1]

        y = torch.cat([torch.zeros(B, max_global_T, D-d, **like(x)), y], -1)

        y = y + self.global_position_encoding[:max_global_T]

        # print patch boundaries
        if c.print_patches > 0 and targets is not None and torch.rand(()) < c.print_patches:
            b0, t0, T0 = 0, 0, 128
            global_ts0, targets0 = global_ts[b0].cpu(), targets[b0].cpu()
            print()
            print(f'TG={global_T[b0]/TG:.0%}, ignored={targets0.eq(-1).float().mean().item():.0%}')
            while t0 < T:
                print(''.join('!' if t_ in global_ts0 else ' ' for t_ in range(t0, t0+T0)) + '|')
                print(util.chrs(targets0[t0:t0+T0]) + '|')
                print(self.dataset_tokenizer.decode(tokens[b0, t0:t0+T0]))
                t0 += T0
            print()

        for block in self.global_blocks:
            y = block(y, log=log)

        if c.patch_method != 'periodic':
            x = torch.stack([
                x0.index_add(0, ts[:Ty0], y0[:Ty0, -d:])
                for x0, ts, Ty0, y0 in zip(x, global_ts, global_T, y, strict=True) ])
        else:
            x = x.index_add(1, torch.arange(0, Tx, P, device=x.device), y[:, :, -d:])
        del y

        for block in self.final_blocks:
            x = block(x, log=log)

        logits = self.logits(x, log=log)

        losses = None
        if targets is not None:
            losses = {}
            ignore_index = -1 if c.patch_method != 'periodic' else None
            losses['cross entropy'] = util.cross_entropy(logits, targets, ignore_index=ignore_index)
            losses['loss'] = losses['cross entropy']

            if c.patch_method != 'periodic':
                losses['global context'] = sum(global_T) / len(global_T)
                losses['ignored fraction'] = (targets == -1).float().mean()
        else:
            if c.patch_method != 'periodic':
                for b, t in stop_gen:
                    # not enough global blocks to continue generation
                    logits[b, t:] = -1e4
                    logits[b, t:, c.BOS] = 1e4

        return logits, losses
