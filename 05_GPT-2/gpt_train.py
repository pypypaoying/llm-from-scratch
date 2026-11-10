import math
import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y=F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50304 # 50257 GPT-2 tokens, padded for matrix alignment
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

        decay_params = [
            p for p in param_dict.values() if p.dim() >= 2
        ]
        nodecay_params = [
            p for p in param_dict.values() if p.dim() < 2
        ]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )

        fused_available = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer

# Single-GPU 100M-token experiment, following the course training loop.
DATA_DIR = Path(__file__).with_name("edu_fineweb100M")
OUT_DIR = Path(__file__).with_name("log_100M")
TRAIN_TOKENS = 100_000_000
total_batch_size = 524288
B, T = 4, 1024
max_lr, min_lr = 6e-4, 6e-5
warmup_steps = 10
max_steps = math.ceil(TRAIN_TOKENS / total_batch_size)
eval_interval = 10
eval_tokens = 131072
use_compile = True


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ratio = min(1.0, (step - warmup_steps) / max(1, max_steps - 1 - warmup_steps))
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B, self.T, self.split = B, T, split
        self.tokens = np.load(DATA_DIR / f"{split}.npy", mmap_mode="r")
        if self.tokens.dtype != np.uint16 or len(self.tokens) < 2:
            raise ValueError(f"Invalid {split} token file")
        self.current_position = 0

    def reset(self):
        self.current_position = 0

    def next_batch(self, token_limit=None):
        requested = min(self.B * self.T, token_limit or self.B * self.T)
        rows = min(self.B, max(1, requested // self.T))
        length = self.T if requested >= self.T else requested
        count = rows * length
        p = self.current_position
        if p + count + 1 > len(self.tokens):
            if self.split == "train":
                raise RuntimeError("Training data exhausted before the token budget")
            p = 0
        # Convert only this batch to int64; keep the dataset on disk.
        buf = torch.from_numpy(np.array(self.tokens[p:p + count + 1], dtype=np.int64))
        self.current_position = p + count
        return buf[:-1].view(rows, length), buf[1:].view(rows, length)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def estimate_loss(model, loader, device, token_limit=eval_tokens):
    was_training = model.training
    model.eval()
    loader.reset()
    count, loss_sum = 0, 0.0
    limit = min(token_limit, len(loader.tokens) - 1)
    try:
        while count < limit:
            x, y = loader.next_batch(limit - count)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            loss_sum += loss.item() * y.numel()
            count += y.numel()
    finally:
        model.train(was_training)
    return loss_sum / count


@torch.no_grad()
def generate(model, device, prompt="The scientist looked at the sky", new_tokens=200):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    was_training = model.training
    model.eval()
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(42)
    try:
        for _ in range(new_tokens):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, _ = model(idx[:, -T:])
            logits = logits[:, -1, :enc.n_vocab].float() / 0.8
            values, indices = torch.topk(logits, 50, dim=-1)
            choice = torch.multinomial(F.softmax(values, dim=-1), 1, generator=rng)
            idx = torch.cat((idx, indices.gather(-1, choice)), dim=1)
    finally:
        model.train(was_training)
    return enc.decode(idx[0].tolist())


def save_checkpoint(raw_model, optimizer, loader, step, tokens_seen, best_val, history, data_id, settings):
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(raw_model.config),
        "step": step, "tokens_seen": tokens_seen,
        "train_position": loader.current_position,
        "best_val": best_val, "history": history,
        "data_id": data_id, "settings": settings,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = OUT_DIR / "last.tmp.pt"
    torch.save(checkpoint, temporary)
    os.replace(temporary, OUT_DIR / "last.pt")


def run(resume=False, stop_after=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run fineweb.py first to prepare train/validation data")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_id = {}
    for split in ("train", "val"):
        description = manifest["splits"][split]
        digest = file_sha256(DATA_DIR / description["file"])
        if digest != description["sha256"]:
            raise ValueError(f"{split} data checksum changed")
        data_id[split] = digest
    settings = {
        "train_tokens": TRAIN_TOKENS, "total_batch_size": total_batch_size,
        "B": B, "T": T, "max_lr": max_lr, "min_lr": min_lr,
        "warmup_steps": warmup_steps, "max_steps": max_steps,
        "eval_interval": eval_interval, "eval_tokens": eval_tokens,
        "compile": use_compile, "precision": "bf16" if device == "cuda" else "fp32",
        "weight_decay": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT_DIR / "last.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError("A checkpoint exists; use --resume to continue it")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError("No last.pt checkpoint to resume")
    train_loader = DataLoaderLite(B, T, "train")
    val_loader = DataLoaderLite(B, T, "val")
    if len(train_loader.tokens) - 1 < TRAIN_TOKENS:
        raise ValueError("Not enough unique training tokens for this experiment")
    raw_model = GPT(GPTConfig()).to(device)
    optimizer = raw_model.configure_optimizers(0.1, max_lr, device)
    step, tokens_seen, best_val, history = 0, 0, float("inf"), []
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint["settings"] != settings or checkpoint["data_id"] != data_id
                or checkpoint["config"] != asdict(raw_model.config)):
            raise ValueError("Resume requires the same experiment settings and data")
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, tokens_seen = checkpoint["step"], checkpoint["tokens_seen"]
        train_loader.current_position = checkpoint["train_position"]
        if train_loader.current_position != tokens_seen:
            raise ValueError("Checkpoint data cursor and processed tokens disagree")
        best_val, history = checkpoint["best_val"], checkpoint["history"]
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint
        print(f"Resumed update {step}, tokens {tokens_seen:,}", flush=True)
    model = torch.compile(raw_model) if use_compile else raw_model
    model.train()
    (OUT_DIR / "config.json").write_text(json.dumps({
        **settings, "model": asdict(raw_model.config), "data": manifest,
        "pytorch": str(torch.__version__), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }, indent=2) + "\n", encoding="utf-8")

    if not history:
        initial_val = estimate_loss(model, val_loader, device)
        best_val = initial_val
        history.append({"step": 0, "tokens": 0, "val_loss": initial_val})
        torch.save({"model": raw_model.state_dict(), "config": asdict(raw_model.config),
                    "step": 0, "tokens_seen": 0, "val_loss": initial_val}, OUT_DIR / "best.pt")
        save_checkpoint(raw_model, optimizer, train_loader, 0, 0, best_val, history, data_id, settings)
        print(f"initial validation loss: {initial_val:.6f}", flush=True)

    started = time.perf_counter()
    while tokens_seen < TRAIN_TOKENS:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        update_tokens = min(total_batch_size, TRAIN_TOKENS - tokens_seen)
        optimizer.zero_grad(set_to_none=True)
        consumed = 0
        loss_accum = torch.zeros((), device=device)
        while consumed < update_tokens:
            x, y = train_loader.next_batch(update_tokens - consumed)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            # Weight the final short microbatch correctly as well.
            weighted_loss = loss * (y.numel() / update_tokens)
            loss_accum += weighted_loss.detach()
            weighted_loss.backward()
            consumed += y.numel()
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0, error_if_nonfinite=True)
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step += 1
        tokens_seen += consumed
        row = {
            "step": step, "tokens": tokens_seen, "train_loss": loss_accum.item(),
            "lr": lr, "grad_norm": norm.item(), "seconds": elapsed,
            "tokens_per_second": consumed / elapsed,
        }
        print(f"step {step:03d}/{max_steps} | tokens {tokens_seen:,}/{TRAIN_TOKENS:,} | "
              f"loss {row['train_loss']:.4f} | lr {lr:.3e} | norm {row['grad_norm']:.2f} | "
              f"{elapsed:.2f}s | {row['tokens_per_second']:.0f} tok/s", flush=True)
        with open(OUT_DIR / "train.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if step % eval_interval == 0 or tokens_seen == TRAIN_TOKENS or step == stop_after:
            val_loss = estimate_loss(model, val_loader, device)
            history.append({**row, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": raw_model.state_dict(), "config": asdict(raw_model.config),
                            "step": step, "tokens_seen": tokens_seen, "val_loss": val_loss},
                           OUT_DIR / "best.pt")
            save_checkpoint(raw_model, optimizer, train_loader, step, tokens_seen, best_val, history, data_id, settings)
            print(f"validation loss: {val_loss:.6f} | checkpoint saved", flush=True)
        if step == stop_after and tokens_seen < TRAIN_TOKENS:
            return {"step": step, "tokens_seen": tokens_seen, "history": history}

    full_val = estimate_loss(model, val_loader, device, len(val_loader.tokens) - 1)
    sample = generate(raw_model, device)
    (OUT_DIR / "sample.txt").write_text(sample + "\n", encoding="utf-8")
    summary = {
        "completed": True, "step": step, "train_tokens": tokens_seen,
        "validation_tokens": len(val_loader.tokens) - 1,
        "full_validation_loss": full_val, "validation_perplexity": math.exp(full_val),
        "best_periodic_validation_loss": best_val, "history": history,
        "seconds_in_this_session": time.perf_counter() - started,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {tokens_seen:,} training tokens | full validation loss {full_val:.6f}", flush=True)
    print(sample, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Course GPT-2: 100M-token single-GPU experiment")
    parser.add_argument("--resume", action="store_true", help="resume log_100M/last.pt")
    run(resume=parser.parse_args().resume)
