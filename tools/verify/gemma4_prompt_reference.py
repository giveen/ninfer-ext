"""Whole-model FP32 reference for a prompt, from the BF16 checkpoint.

Runs all sixty layers and the head for the prompt's tokens and reports the token the reference would
greedily pick next, so the engine's own choice can be compared against an implementation that shares no
code with it. Layer 0 is sliding and layer 5 global here, which is the same alternation the engine's
dispatch sees.

    .venv/bin/python tools/verify/gemma4_prompt_reference.py /tmp/gemma_prompt.i32 [top]

Unlike the single-layer check this cannot attribute a difference to one place: it runs the whole stack
with BF16 weights, while the engine holds FP8 and NVFP4, so a disagreement is as likely to be the
quantization as the engine. Its value is telling the two apart from a shared bug.
"""
import json, math, sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

CKPT = '/mnt/storage/models/gemma/full-31b'
PROMPT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/gemma_prompt.i32'
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 5
INDEX = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
CFG = json.load(open(f'{CKPT}/config.json'))
CFG = CFG.get('text_config', CFG)
EPS = float(CFG['rms_norm_eps'])
H = int(CFG['hidden_size'])
HQ, D, HKV = int(CFG['num_attention_heads']), int(CFG['head_dim']), int(CFG['num_key_value_heads'])
GD, GKHV = int(CFG['global_head_dim']), int(CFG['num_global_key_value_heads'])
LAYERS = int(CFG['num_hidden_layers'])
CAP = float(CFG['final_logit_softcapping'])
FULL_LAYER = [t == 'full_attention' for t in CFG['layer_types']]

def load(name):
    with safe_open(f'{CKPT}/{INDEX[name]}', framework='pt', device='cpu') as f:
        return f.get_tensor(name).to(torch.float32)

def rmsnorm(x, weight, width):
    view = x.reshape(-1, width)
    y = view / torch.sqrt(view.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (y * weight).reshape(-1) if weight is not None else y.reshape(-1)

def rope(x, width, pairs, theta, position):
    view = x.reshape(-1, width)
    i = torch.arange(pairs, dtype=torch.float64)
    angle = position * torch.pow(torch.tensor(theta, dtype=torch.float64), -2.0 * i / width)
    cos, sin = torch.cos(angle).to(torch.float32), torch.sin(angle).to(torch.float32)
    lo, hi = view[:, :pairs].clone(), view[:, width // 2: width // 2 + pairs].clone()
    view[:, :pairs] = lo * cos - hi * sin
    view[:, width // 2: width // 2 + pairs] = lo * sin + hi * cos
    return view.reshape(-1)

ids = np.fromfile(PROMPT, dtype=np.int32).tolist()
TOKENS = len(ids)
embed = load('model.language_model.embed_tokens.weight')
scale = math.sqrt(H)
# The stack's input: the embedding scaled the way the embedding Op scales it, at BF16 resolution.
states = [(embed[i] * scale).to(torch.bfloat16).to(torch.float32) for i in ids]

for layer in range(LAYERS):
    full = FULL_LAYER[layer]
    d, hkv = (GD, GKHV) if full else (D, HKV)
    width = GD if full else D
    group = HQ // hkv
    theta = float(CFG['rope_parameters']['full_attention' if full else 'sliding_attention']['rope_theta'])
    pairs = int(round(GD * 0.25)) // 2 if full else d // 2
    rot = set(range(pairs)) | set(range(width // 2, width // 2 + pairs))
    nonrot = [i for i in range(d) if i not in rot]
    rot_idx = sorted(rot)
    P = f'model.language_model.layers.{layer}'
    W_IN, W_PA = load(f'{P}.input_layernorm.weight'), load(f'{P}.post_attention_layernorm.weight')
    W_PF, W_PO = load(f'{P}.pre_feedforward_layernorm.weight'), load(f'{P}.post_feedforward_layernorm.weight')
    W_Q, W_K = load(f'{P}.self_attn.q_proj.weight'), load(f'{P}.self_attn.k_proj.weight')
    W_QN, W_KN = load(f'{P}.self_attn.q_norm.weight'), load(f'{P}.self_attn.k_norm.weight')
    W_O = load(f'{P}.self_attn.o_proj.weight')
    W_G, W_U = load(f'{P}.mlp.gate_proj.weight'), load(f'{P}.mlp.up_proj.weight')
    W_D = load(f'{P}.mlp.down_proj.weight')
    SCALAR = float(load(f'{P}.layer_scalar').reshape(-1)[0])
    W_V = None if full else load(f'{P}.self_attn.v_proj.weight')

    keys, values, queries = [], [], []
    for t, state in enumerate(states):
        n = rmsnorm(state, W_IN, H)
        q = rmsnorm(W_Q @ n, W_QN, d)
        k = rmsnorm(W_K @ n, W_KN, d)
        v = rmsnorm(W_V @ n if W_V is not None else rmsnorm(W_K @ n, None, d), None, d)
        keys.append(rope(k.reshape(hkv, d), d, pairs, theta, t).reshape(hkv, d))
        values.append(v.reshape(hkv, d))
        queries.append(rope(q.reshape(HQ, d), d, pairs, theta, t).reshape(HQ, d))

    produced = []
    for t in range(TOKENS):
        attended = torch.empty(HQ, d)
        for head in range(HQ):
            kvh = head // group
            if full:
                scores = torch.stack([
                    (queries[t][head][rot_idx] * keys[j][kvh][rot_idx]).sum() +
                    (queries[t][head][nonrot] * W_KN[nonrot] * values[j][kvh][nonrot]).sum()
                    for j in range(t + 1)])
            else:
                scores = torch.stack([(queries[t][head] * keys[j][kvh]).sum() for j in range(t + 1)])
            probability = F.softmax(scores, dim=0)
            attended[head] = sum(probability[j] * values[j][kvh] for j in range(t + 1))
        h = states[t] + rmsnorm(W_O @ attended.reshape(-1), W_PA, H)
        n = rmsnorm(h, W_PF, H)
        product = F.gelu(W_G @ n, approximate='tanh') * (W_U @ n)
        produced.append((h + rmsnorm(W_D @ product, W_PO, H)) * SCALAR)
    states = produced
    if layer % 10 == 0 or layer == LAYERS - 1:
        print(f'  layer {layer} done, last state |h|max {states[-1].abs().max().item():.3f}', flush=True)

final = rmsnorm(states[-1], load('model.language_model.norm.weight'), H)
logits = CAP * torch.tanh((embed @ final) / CAP)
best = torch.topk(logits, TOP)
print('reference next token:', int(best.indices[0]), 'logit', float(best.values[0]))
print('  top:', [(int(i), round(float(v), 3)) for i, v in zip(best.indices, best.values)])
print('  |logits|max', float(logits.abs().max()), 'cap', CAP)
