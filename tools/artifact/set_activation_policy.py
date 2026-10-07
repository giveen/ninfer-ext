"""Change the activation_policy of selected Uses in a single-file v3 artifact, keeping every weight byte.

A Use's stored activation_policy is the permission for the activation precision at that parameter's
mathematical inputs; the Engine selects a route only within the stored permission. Rewriting it is
metadata-only, so enabling a route on an already-downloaded artifact - for example the Q4 W4A8
prefill on the groupwise-int gate/up projections - needs neither a reconversion nor a re-download.
The new directory must fit the existing reserved JSON space, so the payload start does not move.
The result gets a new artifact_id and is verified through the reader before it is published.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import shutil
import uuid

from .framing import HEADER, MAGIC
from .reader import Artifact
from .schema import ACTIVATION_POLICIES, ArtifactError, encode_directory


def set_activation_policy(
    source: Path, target: Path, parameters: list[str], policy: str
) -> int:
    """Set ``policy`` on every Use whose parameter matches one of ``parameters``.

    ``parameters`` are glob patterns matched against the full parameter name with
    :func:`fnmatch.fnmatchcase`, e.g. ``*/mlp/gate``. Returns the number of Uses changed.
    """
    if source.resolve() == target.resolve() or target.exists():
        raise ArtifactError("output must be a new path")
    if policy not in ACTIVATION_POLICIES:
        raise ArtifactError(f"unsupported activation policy {policy!r}")
    with Artifact(source) as artifact:
        if len(artifact.directory.files) != 1:
            raise ArtifactError("only single-file artifacts are supported")
        payload_bytes = artifact.payload_bytes
    with source.open("rb") as stream:
        magic, json_bytes, _ = HEADER.unpack(stream.read(HEADER.size))
        if magic != MAGIC:
            raise ArtifactError("not a v3 artifact entry file")
        description = json.loads(stream.read(json_bytes))
    matched = 0
    changed = 0
    for use in description["uses"]:
        if not any(fnmatchcase(use["parameter"], pattern) for pattern in parameters):
            continue
        matched += 1
        if use.get("activation_policy") == policy:
            continue
        use["activation_policy"] = policy
        changed += 1
    if not matched:
        raise ArtifactError("no Use matched the given parameter patterns")
    if not changed:
        raise ArtifactError(f"matching Uses already use {policy}")
    encoded = encode_directory(description)
    if len(encoded) > json_bytes:
        raise ArtifactError("new directory exceeds the reserved JSON space")
    header = HEADER.pack(MAGIC, json_bytes, uuid.uuid4().bytes)
    temporary = target.with_name(target.name + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as stream:
            stream.write(header)
            stream.write(encoded + b" " * (json_bytes - len(encoded)))
            stream.flush()
            os.fsync(stream.fileno())
        with Artifact(temporary) as artifact:
            if artifact.payload_bytes != payload_bytes:
                raise ArtifactError("patched artifact payload changed")
        with temporary.open("rb") as stream:
            stream.seek(HEADER.size)
            if stream.read(len(encoded)) != encoded:
                raise ArtifactError("patched directory did not read back")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--parameter",
        action="append",
        required=True,
        metavar="GLOB",
        help="Use parameter glob to match, e.g. '*/mlp/gate'; repeatable",
    )
    parser.add_argument("--policy", required=True, choices=sorted(ACTIVATION_POLICIES))
    args = parser.parse_args()
    changed = set_activation_policy(
        args.artifact, args.output, args.parameter, args.policy
    )
    print(f"{changed} Use record(s) set to {args.policy}")


if __name__ == "__main__":
    main()
