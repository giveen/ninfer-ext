"""HuggingFace's own Gemma 4 as the oracle for this model.

The strongest independent check available: not a reimplementation, the reference. It must run on an
interpreter whose `transformers` ships `transformers.models.gemma4` (the 5.x line; the project's
maintainer environment has 4.57.6 and does not), so pass one explicitly, for example:

    /mnt/storage/ninfer/eval/.venv/bin/python tools/verify/gemma4_hf_reference.py

It loads the checkpoint in BF16 on the CPU, which a 31B fits in host RAM and a 32 GB device does not,
and scores a few short texts. Two things it taught, both worth keeping:

- This pipeline's tokenizer does *not* add its beginning-of-sequence token on its own. Without one,
  HF's own model scores the France sentence at NLL 11.99 and wikitext at 14.95 — an implementation
  that adds it is right and stock HF is the odd one out here.
- With it, HF scores the France sentence at 2.837 and wikitext at 8.215, which is what the engine
  gets too (2.509 and 9.070). Real prose genuinely does score that badly on this checkpoint, so the
  engine is faithful and the corpus is what it is.

The ids it scored are written to /tmp/hf_<name>.i32 so the engine can be fed exactly the same
sequence, through ninfer_gemma4_score_test, and the two compared like for like.
"""
import torch, numpy as np
MODEL = '/mnt/storage/models/gemma/full-31b'
torch.set_grad_enabled(False)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.eval()
bos = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids('<bos>')
print('loaded; bos id', bos, flush=True)

TEXTS = {
    'france': ('The capital of France is Paris, the largest city in the country and one of the '
               'most visited places in the world. Millions of people travel there every year.'),
    'wikitext': (' = Robert Boulter = \n\n Robert Boulter is an English film , television and '
                 'theatre actor . He had a guest @-@ starring role on the television series The '
                 'Bill in 2000 .'),
}
for name, text in TEXTS.items():
    ids = [bos] + tok(text).input_ids[:64]
    t = torch.tensor([ids])
    out = model(t, labels=t)
    print(f'{name}: {len(ids)} tokens, HF(bf16, cpu) mean NLL {float(out.loss):.4f}, '
          f'PPL {float(np.exp(float(out.loss))):.2f}', flush=True)
    np.array(ids, dtype=np.int32).tofile(f'/tmp/hf_{name}.i32')
    print(f'  ids: {ids[:12]} ...', flush=True)
