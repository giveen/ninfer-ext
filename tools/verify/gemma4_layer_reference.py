"""Independent FP32 reference for one Gemma 4 decoder layer, checked against the engine.

The engine's own weights are FP8 and NVFP4, so this cannot agree to the last bit. What it rules out is
a structural error: a wrong layout, a missing rotation, a mis-grouped head, a score computed without
the key norm's weight. Those move the cosine far below 1 rather than a few tenths of a percent.

Usage, from the repository root, after dumping a layer from the engine:

    NINFER_GEMMA_ARTIFACT=<artifact> NINFER_GEMMA_LAYER=<layer> NINFER_GEMMA_TOKENS=<tokens> \
        ./build/tests/ninfer_gemma4_layer_test
    .venv/bin/python tools/verify/gemma4_layer_reference.py <layer> <tokens>

Layer 0 is a sliding layer and layer 5 a global one. The reference implements the layer as the
reference engines define it, including the causal attention: the sliding Op's plain dot-product form
with scale 1.0, and the global layers' compact-KV form, where a stored row is the value vector followed
by the rotated key dims and the query carries the key norm's weight on the dims the rotation leaves
alone. It reads BF16 weights, so it needs the original checkpoint beside the converted artifact.
"""
import json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

CKPT = '/mnt/storage/models/gemma/full-31b'
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# The dump fixes the token count, so the reference takes it from there rather than a second argument.
TOKENS = None
INDEX = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
CFG = json.load(open(f'{CKPT}/config.json'))
CFG = CFG.get('text_config', CFG)
EPS = float(CFG['rms_norm_eps'])
H, I = int(CFG['hidden_size']), int(CFG['intermediate_size'])
HQ, D, HKV = int(CFG['num_attention_heads']), int(CFG['head_dim']), int(CFG['num_key_value_heads'])
GD, GKHV = int(CFG['global_head_dim']), int(CFG['num_global_key_value_heads'])
FULL = CFG['layer_types'][LAYER] == 'full_attention'
P = f'model.language_model.layers.{LAYER}'
d, hkv = (GD, GKHV) if FULL else (D, HKV)
group = HQ // hkv
ROPE = CFG['rope_parameters']['full_attention' if FULL else 'sliding_attention']
THETA = float(ROPE['rope_theta'])
# partial_rotary_factor 0.25 over the full 512-wide head rotates 128 dims, which is 64 pairs.
PAIRS = int(round(GD * 0.25)) // 2 if FULL else d // 2
WIDTH = GD if FULL else D
ROT = list(range(PAIRS)) + list(range(WIDTH // 2, WIDTH // 2 + PAIRS))
NONROT = [i for i in range(d) if i not in set(ROT)]

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

if len(sys.argv) > 2 and sys.argv[2] == '--write-input':
    # The layer stack's real input: the embedding of a few tokens, scaled the way the embedding Op
    # scales it. Written as FP32 holding BF16 values, which is what the engine stores.
    WRITE_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    table = load('model.language_model.embed_tokens.weight')
    # The checkpoint's text config carries no embedding_scale; the converter derives the value the
    # embedding Op uses, which is the square root of the hidden size.
    scale = math.sqrt(H)
    ids = torch.arange(1000, 1000 + WRITE_TOKENS)
    rows = (table[ids] * scale).to(torch.bfloat16).to(torch.float32)
    rows.numpy().astype(np.float32).tofile('/tmp/gemma_layer.input.f32')
    print(f'wrote /tmp/gemma_layer.input.f32: {WRITE_TOKENS} tokens, embedding scale {scale:.3f}')
    sys.exit(0)

W_IN = load(f'{P}.input_layernorm.weight')
W_POST_ATTN = load(f'{P}.post_attention_layernorm.weight')
W_PRE_FFN = load(f'{P}.pre_feedforward_layernorm.weight')
W_POST_FFN = load(f'{P}.post_feedforward_layernorm.weight')
W_Q, W_K = load(f'{P}.self_attn.q_proj.weight'), load(f'{P}.self_attn.k_proj.weight')
W_QN, W_KN = load(f'{P}.self_attn.q_norm.weight'), load(f'{P}.self_attn.k_norm.weight')
W_O = load(f'{P}.self_attn.o_proj.weight')
W_G, W_U, W_D = (load(f'{P}.mlp.gate_proj.weight'), load(f'{P}.mlp.up_proj.weight'),
                 load(f'{P}.mlp.down_proj.weight'))
SCALAR = float(load(f'{P}.layer_scalar').reshape(-1)[0])
W_V = load(f'{P}.self_attn.v_proj.weight') if not FULL else None

if not (os.path.exists('/tmp/gemma_layer.in.f32') and os.path.exists('/tmp/gemma_layer.out.f32')):
    sys.exit('no dump in /tmp; run ninfer_gemma4_layer_test first')
inputs = torch.from_numpy(np.fromfile('/tmp/gemma_layer.in.f32', dtype=np.float32).copy())
if TOKENS is None:
    TOKENS = inputs.numel() // H
inputs = inputs.reshape(TOKENS, H)

# Every token's key and value rows first, so a query can attend over the tokens before it.
keys, values, queries = [], [], []
for t in range(TOKENS):
    n = rmsnorm(inputs[t], W_IN, H)
    q = rmsnorm(W_Q @ n, W_QN, d)
    k = rmsnorm(W_K @ n, W_KN, d)
    # Global layers store no value projection: K is V, normalized without a weight.
    v = rmsnorm(W_V @ n if W_V is not None else rmsnorm(W_K @ n, None, d), None, d)
    keys.append(rope(k.reshape(hkv, d), d, PAIRS, THETA, t).reshape(-1))
    values.append(v.reshape(hkv, d))
    queries.append(rope(q.reshape(HQ, d), d, PAIRS, THETA, t).reshape(HQ, d))

outputs = []
for t in range(TOKENS):
    attended = torch.empty(HQ, d)
    for head in range(HQ):
        kvh = head // group
        if FULL:
            # The compact row stores the value vector, then the rotated key dims low run then high run.
            # On the dims the rotation leaves alone the key is the value times the key norm's weight,
            # so the query carries that weight instead.
            scores = []
            for j in range(t + 1):
                rotated = (queries[t][head][ROT] * keys[j].reshape(hkv, d)[kvh][ROT]).sum()
                plain = (queries[t][head][NONROT] * W_KN[NONROT] *
                         values[j][kvh][NONROT]).sum()
                scores.append(rotated + plain)
            probability = F.softmax(torch.stack(scores), dim=0)
            attended[head] = sum(probability[j] * values[j][kvh] for j in range(t + 1))
        else:
            scores = torch.stack([(queries[t][head] * keys[j].reshape(hkv, d)[kvh]).sum()
                                  for j in range(t + 1)])
            probability = F.softmax(scores, dim=0)
            attended[head] = sum(probability[j] * values[j][kvh] for j in range(t + 1))
    h = inputs[t] + rmsnorm(W_O @ attended.reshape(-1), W_POST_ATTN, H)
    n = rmsnorm(h, W_PRE_FFN, H)
    product = F.gelu(W_G @ n, approximate='tanh') * (W_U @ n)
    h = h + rmsnorm(W_D @ product, W_POST_FFN, H)
    outputs.append(h * SCALAR)

reference = torch.stack(outputs).reshape(-1)
engine = torch.from_numpy(np.fromfile('/tmp/gemma_layer.out.f32', dtype=np.float32).copy())
for t in range(TOKENS):
    a, b = reference[t * H:(t + 1) * H], engine[t * H:(t + 1) * H]
    cos = F.cosine_similarity(a[None], b[None]).item()
    rel = ((a - b).norm() / b.norm()).item()
    print(f'layer {LAYER} ({"global" if FULL else "sliding"}) token {t}: cosine {cos:.6f}, '
          f'relative L2 {rel:.4f}, max abs diff {torch.abs(a - b).max().item():.4f}')
# Optional head check. The head reads the engine's own layer output, so this isolates the final norm,
# the tied output projection and the soft cap from any error the layers themselves contributed.
if os.path.exists('/tmp/gemma_head.out.f32'):
    engine_logits = torch.from_numpy(np.fromfile('/tmp/gemma_head.out.f32', dtype=np.float32).copy())
    W_FINAL = load('model.language_model.norm.weight')
    W_EMB = load('model.language_model.embed_tokens.weight')
    CAP = float(CFG['final_logit_softcapping'])
    n = rmsnorm(engine[(TOKENS - 1) * H:TOKENS * H], W_FINAL, H)
    # The embedding scale applies at the input; the tied output projection uses the matrix unscaled.
    logits = CAP * torch.tanh((W_EMB @ n) / CAP)
    cos = F.cosine_similarity(logits[None], engine_logits[None]).item()
    rel = ((logits - engine_logits).norm() / engine_logits.norm()).item()
    agree = int(logits.argmax()) == int(engine_logits.argmax())
    print(f'head ({"global" if FULL else "sliding"} layer): cosine {cos:.6f}, relative L2 {rel:.4f}, '
          f'|logits|max {engine_logits.abs().max().item():.3f}, cap {CAP}')
    print(f'  argmax {int(logits.argmax())} (reference) vs {int(engine_logits.argmax())} (engine): '
          f'{"agree" if agree else "DIFFER"}')
    print('  HEAD VERDICT:', 'structural agreement' if cos > 0.99 else 'MISMATCH — investigate')

print('  VERDICT:', 'structural agreement' if min(
    F.cosine_similarity(reference[t * H:(t + 1) * H][None], engine[t * H:(t + 1) * H][None]).item()
    for t in range(TOKENS)) > 0.99 else 'MISMATCH — investigate')
