"""Attach a compact draft head to an artifact: the shortlisted rows of `text/output_head`.

The draft head is the output head restricted to the ids a shortlist keeps, stored in the same
format and layout as the source so the draft projection is the same Op over fewer rows. Every row
is copied as encoded bytes — nothing is decoded or requantized — and the id of each retained row
travels with the object, so the draft's argmax maps back to a global token id.

The result is a new artifact; the source is never modified.

    tools/draft_vocab/attach.py --model in.ninfer --vocab draft_vocab.bin --out out.ninfer
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from tools.artifact.layouts import row_split_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter

HEAD_BINDING = 'text/output_head'
HEAD_PARAMETER = 'text/draft_head'
IDS_PARAMETER = 'text/draft_head_ids'
SUPPORTED_FORMATS = ('q6_g64_fp16', 'q8_g32_fp16')


def plan(base: Artifact, vocab_path: Path):
    directory = base.directory
    if HEAD_PARAMETER in directory.bindings:
        raise ArtifactError('artifact already carries a draft head')
    binding = directory.bindings.get(HEAD_BINDING)
    if not binding or 'object' not in binding:
        raise ArtifactError(f'{HEAD_BINDING} is not a single-object binding')
    head = base.by_id[binding['object']]
    if not isinstance(head, TensorObject):
        raise ArtifactError('output head binding does not name a tensor')
    if head.format not in SUPPORTED_FORMATS or head.layout != 'row_split_k128_v1':
        raise ArtifactError(f'unsupported head storage {head.format}/{head.layout}')

    ids = np.fromfile(vocab_path, dtype='<i4')
    if ids.size == 0:
        raise ArtifactError('empty shortlist')
    if np.any(np.diff(ids) <= 0):
        raise ArtifactError('shortlist ids must be strictly ascending')
    if ids[0] < 0 or ids[-1] >= head.shape[0]:
        raise ArtifactError('shortlist id outside the head')

    source = row_split_geometry(head.format, head.shape)
    compact = row_split_geometry(head.format, (int(ids.size), head.shape[1]))
    names = {'head': 'draft/draft_head', 'ids': 'draft/draft_head_ids'}

    specs = []
    for obj in directory.objects:
        if isinstance(obj, TensorObject):
            specs.append(TensorSpec(obj.id, obj.shape, obj.format, obj.layout, obj.divisors,
                                    obj.bitrate_half_bits))
        else:
            from tools.artifact.schema import ResourceSpec

            specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
    specs.append(TensorSpec(names['head'], (int(ids.size), head.shape[1]), head.format,
                            head.layout))
    specs.append(TensorSpec(names['ids'], (int(ids.size),), 'int32', 'contiguous_le_v1'))

    bindings = deepcopy(directory.bindings)
    bindings[HEAD_PARAMETER] = {'object': names['head']}
    bindings[IDS_PARAMETER] = {'object': names['ids']}
    uses = deepcopy(list(directory.uses))
    uses.append({'parameter': HEAD_PARAMETER, 'input': 'mtp/final_hidden',
                 'activation_policy': 'A16Only'})

    return {
        'head': head,
        'source': source,
        'compact': compact,
        'ids': ids,
        'names': names,
        'specs': specs,
        'bindings': bindings,
        'uses': uses,
    }


def build_payload(base_payload, source, compact, ids):
    planes = [
        (source.base_offset, source.base_row_bytes, compact.base_offset),
        (source.high_offset, source.high_row_bytes, compact.high_offset),
        (source.scale_offset, source.scale_row_bytes, compact.scale_offset),
    ]
    payload = bytearray(compact.payload_bytes)
    for src_offset, row_bytes, dst_offset in planes:
        if row_bytes == 0:
            continue
        plane = bytearray(row_bytes * int(ids.size))
        view = memoryview(base_payload)
        for r, token_id in enumerate(ids):
            start = src_offset + int(token_id) * row_bytes
            plane[r * row_bytes:(r + 1) * row_bytes] = view[start:start + row_bytes]
        payload[dst_offset:dst_offset + len(plane)] = plane
    return bytes(payload)


def attach(model_path, vocab_path, output_path, *, progress=print):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.monotonic()
    with Artifact(model_path) as base:
        layout = plan(base, Path(vocab_path))
        head = layout['head']
        compact = layout['compact']
        ids = layout['ids']
        progress(f'head {head.shape[0]}x{head.shape[1]} {head.format} -> '
                 f'{ids.size} rows, {compact.payload_bytes} bytes')
        payload = build_payload(base.read_object(head.id), layout['source'], compact, ids)
        ids_payload = ids.astype('<i4').tobytes()

        required = base.directory.payload_bytes + compact.payload_bytes + 64 * 1024**2
        if shutil.disk_usage(output_path.parent).free < required:
            raise ArtifactError('insufficient output disk space')

        provenance = deepcopy(base.directory.provenance)
        provenance.setdefault('draft_head', []).append({
            'source_object': head.id,
            'source_artifact_id': base.artifact_id.hex(),
            'rows': int(ids.size),
            'vocab': str(vocab_path),
            'vocab_sha256': hashlib.sha256(Path(vocab_path).read_bytes()).hexdigest(),
            'operation': 'encoded-row-gather',
        })

        with ArtifactWriter(
            output_path,
            layout['specs'],
            components=deepcopy(base.directory.components),
            bindings=layout['bindings'],
            uses=layout['uses'],
            metadata=deepcopy(base.directory.metadata),
            provenance=provenance,
        ) as writer:
            for index, obj in enumerate(base.directory.objects):
                writer.write_object(obj.id, base.iter_object(obj.id))
                if (index + 1) % 50 == 0:
                    progress(f'copied {index + 1}/{len(base.directory.objects)} objects')
            writer.write_object(layout['names']['head'], [payload])
            writer.write_object(layout['names']['ids'], [ids_payload])

    report = {
        'model': str(model_path),
        'vocab': str(vocab_path),
        'output': str(output_path),
        'rows': int(ids.size),
        'head_bytes': compact.payload_bytes,
        'ids_bytes': len(ids_payload),
        'seconds': round(time.monotonic() - started, 1),
    }
    Path(str(output_path) + '.draft_head.json').write_text(json.dumps(report, indent=1) + '\n')
    progress(json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--vocab', required=True, help='shortlist .bin from build.py')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        attach(args.model, args.vocab, args.out)
    except (ArtifactError, FileExistsError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
