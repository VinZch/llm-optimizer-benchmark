"""Persistent-state codecs and explicitly ordered AdamW adapter (RNE only)."""
import copy
import math
import torch

CODECS = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}

def encode(x, codec, writes=1):
    if x.dtype != torch.float32 or codec not in CODECS:
        raise ValueError("codec requires FP32 input and a supported format")
    torch._assert_async(torch.isfinite(x).all(), "nonfinite state")
    effective = codec if x.numel() >= 4096 else "fp32"
    result = dict(schema=1, codec=effective, requested=codec, shape=tuple(x.shape), writes=writes)
    if effective != "fp8":
        result.update(payload=x.to(CODECS[effective]).clone(), scales=None)
        return result
    n = x.numel()
    flat = x.contiguous().view(-1)
    pad = (-n) % 128
    blocks = torch.nn.functional.pad(flat, (0, pad)).view(-1, 128)
    amax = blocks.abs().amax(dim=1)
    exact = amax.double() / 448.0
    scale = exact.float()
    scale = torch.where(scale.double() < exact, torch.nextafter(scale, torch.full_like(scale, math.inf)), scale)
    scale = torch.where(amax == 0, torch.ones_like(scale), scale)
    normalized = blocks / scale[:, None]
    normalized = torch.where(amax[:,None] == 0, torch.zeros_like(normalized), normalized)
    # No silent saturation: upward scales must keep every normalized value finite.
    torch._assert_async((normalized.abs() <= 448).all(), "FP8 overflow")
    result.update(payload=normalized.to(torch.float8_e4m3fn).flatten()[:n].clone(), scales=scale)
    return result

def decode(state):
    if state["schema"] != 1:
        raise ValueError("unsupported state schema")
    x = state["payload"].float()
    if state["codec"] == "fp8":
        n = x.numel()
        x = (torch.nn.functional.pad(x, (0, (-n) % 128)).view(-1, 128) * state["scales"][:, None]).flatten()[:n]
    return x.view(state["shape"])

def stored_bytes(state):
    return sum(t.numel() * t.element_size() for t in (state["payload"], state["scales"]) if t is not None)

class QuantizedAdamW(torch.optim.Optimizer):
    """Current update uses unencoded moments; encode only after their final use."""
    def __init__(self, params, lr=5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, m_codec="fp32", v_codec="fp32"):
        if lr < 0 or eps < 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("invalid AdamW hyperparameters")
        if m_codec not in CODECS or v_codec not in CODECS:
            raise ValueError("unsupported codec")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.m_codec, self.v_codec = m_codec, v_codec

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse or p.dtype != torch.float32 or g.dtype != torch.float32:
                    raise ValueError("adapter supports dense FP32 parameters and gradients")
                s = self.state[p]
                if not s:
                    s["step"] = 0
                    m, v = torch.zeros_like(p), torch.zeros_like(p)
                else:
                    m, v = decode(s["exp_avg"]), decode(s["exp_avg_sq"])
                s["step"] += 1
                step = s["step"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = v.sqrt().div_(math.sqrt(1 - b2 ** step)).add_(group["eps"])
                p.addcdiv_(m, denom, value=-group["lr"] / (1 - b1 ** step))
                s["exp_avg"] = encode(m, self.m_codec, step)
                s["exp_avg_sq"] = encode(v, self.v_codec, step)
                del m, v, denom
        return loss

    def state_dict(self):
        result = super().state_dict()
        result["quantization"] = dict(schema=1, m=self.m_codec, v=self.v_codec)
        return result

    def load_state_dict(self, state_dict):
        # Optimizer's default loader casts FP8 payloads to parameter dtype.
        q = state_dict.get("quantization")
        if q != dict(schema=1, m=self.m_codec, v=self.v_codec):
            raise ValueError("checkpoint codec/schema mismatch")
        saved = copy.deepcopy(state_dict)
        groups = saved["param_groups"]
        if len(groups) != len(self.param_groups) or any(len(a["params"]) != len(b["params"]) for a,b in zip(groups,self.param_groups)):
            raise ValueError("checkpoint parameter groups mismatch")
        mapping = {idx:p for a,b in zip(groups,self.param_groups) for idx,p in zip(a["params"],b["params"])}
        self.state.clear()
        for idx, s in saved["state"].items():
            p = mapping[idx]
            for key in ("exp_avg", "exp_avg_sq"):
                encoded = s[key]
                if tuple(p.shape) != tuple(encoded["shape"]):
                    raise ValueError("checkpoint shape mismatch")
                encoded["payload"] = encoded["payload"].to(device=p.device)
                if encoded["scales"] is not None:
                    encoded["scales"] = encoded["scales"].to(device=p.device)
            self.state[p] = s
        for target, source in zip(self.param_groups, groups):
            target.update({k:v for k,v in source.items() if k != "params"})
