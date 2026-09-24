"""Replace the trailing resource object of a single-file v3 artifact, keeping every weight byte.

The replaced resource must be the last object in the payload, so no other object moves. The new
directory must fit the existing reserved JSON space, so the payload start does not move either.
The result gets a new artifact_id and is verified through the reader before it is published.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import uuid

from .framing import HEADER, MAGIC
from .reader import Artifact
from .schema import ArtifactError, encode_directory


def replace_resource(source: Path, target: Path, resource_id: str, data: bytes) -> None:
    if source.resolve() == target.resolve() or target.exists():
        raise ArtifactError("output must be a new path")
    with Artifact(source) as artifact:
        if len(artifact.directory.files) != 1:
            raise ArtifactError("only single-file artifacts are supported")
        payload_bytes = artifact.payload_bytes
    with source.open("rb") as stream:
        magic, json_bytes, _ = HEADER.unpack(stream.read(HEADER.size))
        if magic != MAGIC:
            raise ArtifactError("not a v3 artifact entry file")
        description = json.loads(stream.read(json_bytes))
    objects = description["objects"]
    last = max(objects, key=lambda obj: obj["offset"])
    if last["id"] != resource_id or last.get("kind") != "resource":
        raise ArtifactError(f"{resource_id} is not the trailing resource object")
    if last["offset"] + last["bytes"] != payload_bytes:
        raise ArtifactError("trailing resource does not end the payload")
    delta = len(data) - last["bytes"]
    last["bytes"] = len(data)
    description["files"] = [{"path": None, "payload_bytes": payload_bytes + delta}]
    encoded = encode_directory(description)
    if len(encoded) > json_bytes:
        raise ArtifactError("new directory exceeds the reserved JSON space")
    header = HEADER.pack(MAGIC, json_bytes, uuid.uuid4().bytes)
    entry_start = HEADER.size + json_bytes
    if entry_start % 4096:
        entry_start += 4096 - entry_start % 4096
    temporary = target.with_name(target.name + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as stream:
            stream.write(header)
            stream.write(encoded + b" " * (json_bytes - len(encoded)))
            stream.seek(entry_start + last["offset"])
            stream.write(data)
            stream.truncate(entry_start + last["offset"] + len(data))
            stream.flush()
            os.fsync(stream.fileno())
        with Artifact(temporary) as artifact:
            record = next(o for o in artifact.objects if o.id == resource_id)
            if artifact.payload_bytes != payload_bytes + delta or record.bytes != len(data):
                raise ArtifactError("replaced artifact directory is inconsistent")
        with temporary.open("rb") as stream:
            stream.seek(entry_start + last["offset"])
            if stream.read(len(data) + 1) != data:
                raise ArtifactError("replaced resource did not read back")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--resource",
        required=True,
        metavar="ID=PATH",
        help="object id and replacement file, e.g. frontend/chat_template.jinja=qwen.jinja",
    )
    args = parser.parse_args()
    resource_id, _, path = args.resource.partition("=")
    if not resource_id or not path:
        parser.error("--resource must be ID=PATH")
    replace_resource(args.artifact, args.output, resource_id, Path(path).read_bytes())


if __name__ == "__main__":
    main()
