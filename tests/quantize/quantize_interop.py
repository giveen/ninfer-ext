"""Exercise the Python artifact writer through the C++ quantizer's parameter enumeration.

Host-only: `--list` returns before any device call, so this runs without a GPU.
"""

from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter


def main() -> int:
    executable = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="ninfer-quantize-interop-") as temporary:
        path = Path(temporary) / "model.ninfer"
        data = bytes((index * 11 + 3) % 251 for index in range(128 * 128 * 2))
        with ArtifactWriter(
            path,
            [TensorSpec("weight/000000", (128, 128), "bf16", "contiguous_le_v1")],
            components={"text": {"config": {}}},
            bindings={
                # One executable projection and two bindings the quantizer must ignore: an
                # embedding without a Use, and a projection whose K is not 128-aligned.
                "text/layers/0/mlp/gate": {"object": "weight/000000"},
                "text/token_embedding": {"object": "weight/000000"},
            },
            uses=[{"parameter": "text/layers/0/mlp/gate", "input": "text/input"}],
        ) as writer:
            writer.write_object("weight/000000", data)
        result = subprocess.run(
            [executable, str(path), "--list"], capture_output=True, text=True
        )
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        if "eligible parents: 1" not in result.stdout:
            print(result.stdout, file=sys.stderr)
            return 1
        if "text/layers/0/mlp/gate" not in result.stdout:
            print(result.stdout, file=sys.stderr)
            return 1
        if "text/token_embedding" in result.stdout:
            print(result.stdout, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
