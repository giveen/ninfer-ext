"""The Nemotron-H tokenizer against the reference pipeline over a real corpus.

`ninfer_nemotron_h_tokenizer_test` pins a fixed case list against the same oracle; this walks a whole
corpus through both and requires the id streams to be identical, which is the only way to see a split
that diverges once in a few hundred thousand tokens.

Two independent readers meet here: the C++ probe takes the tokenizer out of the artifact through the
loader, and this file takes it out of the artifact through `tools.artifact.reader` -- the probe's own
copy is hashed against this one, so a probe wired to another tokenizer cannot pass.

    NINFER_NEMOTRON_ARTIFACT=out/nemotron_h_74b_nvfp4.ninfer \\
    NINFER_NEMOTRON_CORPUS=/mnt/storage/tmp/flash-wikitext-00.txt \\
    python3 -m pytest tests/models/nemotron_h/test_tokenizer_reference.py

`NINFER_TOKEN_PROBE` names the probe binary; ctest sets it from the built target. The corpus is a
UTF-8 text file; the source checkpoint (NINFER_NEMOTRON_SOURCE, default
`/mnt/storage/models/nemo/74B`) is read too, to check that the artifact's normalized tokenizer.json
tokenizes it exactly as the checkpoint's own copy does.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from tokenizers import Tokenizer
except ImportError:
    raise SystemExit(77)

# The repository root, so `tools.*` imports resolve when ctest runs this file directly as a script.
# Other interop scripts under tests/ do the same, and without it a missing environment variable cannot
# even reach the skip in setUp -- the import fails first, and the test reports a crash instead of a skip.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.artifact.reader import Artifact

ARTIFACT_ENV = "NINFER_NEMOTRON_ARTIFACT"
CORPUS_ENV = "NINFER_NEMOTRON_CORPUS"
SOURCE_ENV = "NINFER_NEMOTRON_SOURCE"
DEFAULT_SOURCE = "/mnt/storage/models/nemo/74B"
PROBE = Path(os.environ.get("NINFER_TOKEN_PROBE", ROOT / "build/tests/ninfer_nemotron_h_tokenizer_test"))

# A missing input is a skip, and this has to happen at module level. unittest catches SystemExit from a
# test body or setUp (it uses a bare except), so raising 77 there would be reported as an error rather
# than a skip; and unittest's own skipTest leaves the process exiting 0, which ctest reads as a test that
# ran nothing having passed. Exiting 77 here is what the registration's SKIP_RETURN_CODE maps to, and it
# matches the C++ tests in this directory. Under pytest the same condition is a module-level skip.
_missing = next(
    (f"{name} is not set" for name in (ARTIFACT_ENV, CORPUS_ENV) if os.environ.get(name) is None),
    None,
)
if _missing is None and not PROBE.exists():
    _missing = f"the probe binary is not built: {PROBE}"
if _missing is not None:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(_missing, allow_module_level=True)
    print(f"SKIP: {_missing}", file=sys.stderr)
    raise SystemExit(77)


def artifact_tokenizer(artifact_path: str) -> str:
    with Artifact(artifact_path) as artifact:
        return artifact.read_object("resource/text/tokenizer.json").decode("utf-8")


def probe_ids(corpus_path: str, tokenizer_out: str) -> list[int]:
    """The C++ tokenizer's ids for one corpus, and the tokenizer.json it loaded."""
    result = subprocess.run(
        [str(PROBE), "--print-ids", corpus_path, tokenizer_out],
        capture_output=True, text=True, check=True, timeout=1800,
    )
    return json.loads(result.stdout)["ids"]


def first_divergence(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


class TokenizerReference(unittest.TestCase):
    def test_corpus_ids_match_the_reference_pipeline(self):
        artifact_path = os.environ[ARTIFACT_ENV]
        corpus_path = os.environ[CORPUS_ENV]
        reference_json = artifact_tokenizer(artifact_path)

        with tempfile.TemporaryDirectory() as workdir:
            loaded_json = os.path.join(workdir, "tokenizer.json")
            ids = probe_ids(corpus_path, loaded_json)
            with open(loaded_json, "rb") as source:
                digest = hashlib.sha256(source.read()).hexdigest()
        self.assertEqual(hashlib.sha256(reference_json.encode("utf-8")).hexdigest(), digest,
                         "the probe tokenized with another tokenizer.json")

        with open(corpus_path, encoding="utf-8") as source:
            corpus = source.read()
        reference = Tokenizer.from_str(reference_json)
        want = reference.encode(corpus, add_special_tokens=False).ids

        print(f"corpus {corpus_path}: {len(corpus)} bytes, {len(ids)} ids")
        if ids != want:
            index = first_divergence(ids, want)
            window = max(0, index - 8)
            got_span = reference.decode(ids[window:index + 8])
            want_span = reference.decode(want[window:index + 8])
            self.fail(
                f"first divergence at id {index} of {len(want)}:\n"
                f"  got  {ids[window:index + 8]}\n"
                f"  want {want[window:index + 8]}\n"
                f"  got  text {got_span!r}\n"
                f"  want text {want_span!r}"
            )
        self.assertEqual(len(ids), len(want))

    def test_artifact_copy_matches_the_checkpoint(self):
        """The artifact's normalized tokenizer.json is the checkpoint's, token for token."""
        source_dir = os.environ.get(SOURCE_ENV, DEFAULT_SOURCE)
        source_path = Path(source_dir) / "tokenizer.json"
        if not source_path.exists():
            self.skipTest(f"the source checkpoint is not present: {source_path}")
        corpus_path = os.environ[CORPUS_ENV]

        with open(corpus_path, encoding="utf-8") as source:
            corpus = source.read()
        artifact_ids = Tokenizer.from_str(artifact_tokenizer(os.environ[ARTIFACT_ENV])).encode(
            corpus, add_special_tokens=False
        ).ids
        checkpoint_ids = Tokenizer.from_str(source_path.read_text(encoding="utf-8")).encode(
            corpus, add_special_tokens=False
        ).ids
        print(f"artifact copy {len(artifact_ids)} ids, checkpoint copy {len(checkpoint_ids)} ids")
        self.assertEqual(artifact_ids, checkpoint_ids,
                         "the artifact's tokenizer.json tokenizes the corpus differently from the "
                         "checkpoint's own copy")


if __name__ == "__main__":
    unittest.main()
