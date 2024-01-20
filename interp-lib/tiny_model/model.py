import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import os
import torch.distributions as dists
from tiny_model.tokenization import enc, dec

current_file = os.path.abspath(__file__)
current_dir = '/'.join(current_file.split('/')[:-1])

def recursively_name_modules(module):
    for name, child in module.named_children():
        child.name = f"{module.name}.{name}" if hasattr(module, 'name') else name
        recursively_name_modules(child)

class HookPoint(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x

    def __repr__(self):
        if self.name is not None:
            return f"HookPoint('{self.name}')"
        else:
            return 'HookPoint()'


class Attention(nn.Module):
    def __init__(self, n_heads, d_model, d_head, max_seq_len):
        super().__init__()
        self.Q = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.K = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.V = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.O = nn.Linear(d_head * n_heads, d_model)

        # https://pi-tau.github.io/posts/transformer/
        nn.init.normal_(self.Q.weight, std=np.sqrt(2 / (d_model + d_head)))
        nn.init.normal_(self.K.weight, std=np.sqrt(2 / (d_model + d_head)))
        nn.init.zeros_(self.O.bias)

        self.attn_mask = nn.Parameter(
            torch.triu(torch.ones(max_seq_len, max_seq_len) * (-np.inf), 1),
            requires_grad=False,
        )
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_heads, max_seq_len))

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.max_seq_len = max_seq_len

        self.qs = HookPoint()
        self.ks = HookPoint()
        self.vs = HookPoint()
        self.head_writeouts = HookPoint()
        self.catted_head_writeouts = HookPoint()
        self.attn_writeouts = HookPoint()

    @property
    def Wq(self):
        return einops.rearrange(self.Q.weight.detach(), "d (h k) -> h d k")

    @property
    def Wk(self):
        return einops.rearrange(self.K.weight.detach(), "d (h k) -> h d k")

    @property
    def Wv(self):
        return einops.rearrange(self.V.weight.detach(), "d (h k) -> h d k")

    @property
    def Wo(self):
        return self.O.weight.detach()

    def pos_bias(self):
        res = (
            torch.zeros(
                self.rel_pos_bias.shape[0],
                self.rel_pos_bias.shape[-1],
                self.rel_pos_bias.shape[-1],
            )
            .to(self.O.bias.device)
            .to(self.O.bias.dtype)
        )

        for i in range(self.rel_pos_bias.shape[-1]):
            res[:, i, : (i + 1)] = self.rel_pos_bias[:, -(i + 1) :]

        return res

    def forward(self, x):
        B, T, _ = x.shape

        q, k, v = self.Q(x), self.K(x), self.V(x)

        qs = einops.rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        qs = self.qs(qs) # hookpoint

        ks = einops.rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        ks = self.ks(ks) # hookpoint

        vs = einops.rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)
        
        vs = self.vs(vs) # hookpoint

        qk_logits = (
            torch.einsum("bhqd,bhkd->bhqk", qs, ks)
            + self.attn_mask[None, None, :T, :T]
            + self.pos_bias()[None, :, :T, :T]
        )
        qk_logits /= self.d_head

        # # softmax_1
        # qk_exps = torch.exp(qk_logits-qk_logits.max(dim=-1).values.unsqueeze(-1))
        # qk_probs = qk_exps/(qk_exps.sum(dim=-1, keepdim=True)+1)

        qk_probs = F.softmax(qk_logits, dim=-1)

        head_writeouts = torch.einsum("b h q k, b h k d -> b h q d", qk_probs, vs)
        
        head_writeouts = self.head_writeouts(head_writeouts) #hookpoint

        catted_head_writeouts = einops.rearrange(head_writeouts, "b h q d -> b q (h d)")

        catted_head_writeouts = self.catted_head_writeouts(catted_head_writeouts) #hookpoint
        
        writeouts = self.O(catted_head_writeouts)
        
        writeouts = self.attn_writeouts(writeouts) #hookpoint

        return writeouts


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.read_in = nn.Linear(d_model, d_mlp)
        self.act = nn.ReLU()
        self.write_out = nn.Linear(d_mlp, d_model)

        self.d_model = d_model
        self.d_mlp = d_mlp

    def forward(self, x):
        preacts = self.read_in(x)
        acts = self.act(preacts)
        return self.write_out(acts)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0, "n_heads must divide d_model"
        d_head = d_model // n_heads

        
        self.attn = Attention(
            n_heads=n_heads, d_model=d_model, d_head=d_head, max_seq_len=max_seq_len
        )
        self.mlp = MLP(d_model=d_model, d_mlp=4 * d_model)

        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

    def forward(self, x):
        x = self.attn(x) + x
        x = self.mlp(x) + x
        return x


class TinyModel(nn.Module):
    def __init__(self, d_model=768, n_layers=4, n_heads=16, max_seq_len=256, vocab_size=10_000, from_pretrained='ts_L-11_E2.pt'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.torso = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # recursively_name_modules(self)

        if isinstance(from_pretrained, str):
            self.load_state_dict(get_state_dict(from_pretrained))
        else:
            assert from_pretrained is False, 'from_pretrained kwarg must be False or a string specifying model'
        
    def forward(self, tok_ids):
        x = self.embed(tok_ids)
        x = self.torso(x)
        logits = self.lm_head(x)
        return F.log_softmax(logits, dim=-1)

    def generate(self, prompt, n_toks=50, temperature=0.8):
        assert temperature >= 0.0
        toks = enc(prompt).to(self.lm_head.weight.device)

        for _ in range(n_toks):
            with torch.no_grad():
                logprobs = self.forward(toks)[0,-1]
                if temperature == 0:
                    next_tok  = logprobs.argmax().item()
                else:
                    next_tok = dists.Categorical(logits=logprobs*(1/temperature)).sample()
            toks = torch.cat((toks, torch.tensor([[next_tok]]).to(toks.device)), dim=-1)
            if next_tok == enc('[END]', add_begin=False).item():
                break
        return dec(toks[:,1:])[0]
            

def get_state_dict(model_fname='ts_L-11_E2.pt'):
    assert model_fname in os.listdir(current_dir+'/models'), f"{os.listdir(current_dir+'/models')}"
    return torch.load(current_dir+'/models/'+model_fname)