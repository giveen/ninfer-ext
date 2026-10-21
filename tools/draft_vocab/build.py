#!/usr/bin/env python3
"""Build a draft-vocabulary shortlist: the token ids a draft head keeps.

The shortlist restricts only the draft head's argmax; verification still runs the full head, so a
token outside the list can still be accepted, it just cannot be proposed. Rank ids by how often a
tokenizer emits them over a corpus and keep the most frequent `--rows`.

Output is little-endian int32 token ids in ascending order (so a row gather at load is coalesced),
plus a JSON sidecar carrying the corpus, the sizes, and the held-out token coverage.

    tools/draft_vocab/build.py --tokenizer /tmp/fn-tok --corpus docs --corpus src --rows 32768 \
        --out models/flash_next_draft_vocab.bin

A held-out fraction of the files is counted separately and never contributes ids, so the printed
coverage is measured against text the list was not built from.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

TEXT_SUFFIXES = ('.md', '.py', '.cpp', '.h', '.cu', '.cuh', '.hpp', '.txt', '.jinja', '.json')


def iter_files(roots):
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {'.git', 'build', '__pycache__'}]
            for name in sorted(filenames):
                if name.endswith(TEXT_SUFFIXES):
                    yield os.path.join(dirpath, name)


def count(tokenizer, paths, counter, positions):
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
                text = handle.read()
        except OSError:
            continue
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        counter.update(ids)
        positions[0] += len(ids)
        positions[1] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokenizer', required=True, help='HF id or local tokenizer directory')
    parser.add_argument('--corpus', action='append', required=True, help='file or directory')
    parser.add_argument('--rows', type=int, default=32768, help='shortlist size')
    parser.add_argument('--holdout', type=float, default=0.1, help='fraction of files held out')
    parser.add_argument('--out', required=True, help='shortlist .bin path')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    files = list(iter_files(args.corpus))
    if not files:
        raise SystemExit('no corpus files')
    rng = random.Random(args.seed)
    rng.shuffle(files)
    held = max(1, int(len(files) * args.holdout))
    train, test = files[held:], files[:held]

    counts = Counter()
    train_positions = [0, 0]
    count(tokenizer, train, counts, train_positions)

    ranked = [token_id for token_id, _ in counts.most_common(args.rows)]
    keep = sorted(ranked)
    if not keep:
        raise SystemExit('empty vocabulary')
    vocab_size = max(counts) + 1
    if keep[-1] >= vocab_size:
        raise SystemExit(f'id {keep[-1]} exceeds the observed vocabulary')

    held_counts = Counter()
    test_positions = [0, 0]
    count(tokenizer, test, held_counts, test_positions)
    covered = sum(n for token_id, n in held_counts.items() if token_id in set(keep))

    with open(args.out, 'wb') as out:
        out.write(b''.join(int(token_id).to_bytes(4, 'little', signed=False) for token_id in keep))

    report = {
        'rows': len(keep),
        'vocab_observed': vocab_size,
        'train_files': len(train),
        'train_tokens': train_positions[0],
        'holdout_files': len(test),
        'holdout_tokens': test_positions[0],
        'holdout_token_coverage': (covered / test_positions[0]) if test_positions[0] else 0.0,
        'min_id': keep[0],
        'max_id': keep[-1],
        'tokenizer': args.tokenizer,
        'corpus': args.corpus,
        'bytes': len(keep) * 4,
    }
    with open(args.out + '.json', 'w') as meta:
        json.dump(report, meta, indent=1)
        meta.write('\n')
    print(json.dumps(report, indent=1))
    print(f'wrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
