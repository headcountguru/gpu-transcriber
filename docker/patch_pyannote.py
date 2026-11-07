"""Build-time patch: rewrite ``use_auth_token`` to ``token`` across the installed pyannote packages.

huggingface_hub removed the ``use_auth_token`` argument; pyannote.audio 3.4.0 still passes it.
Fails the build (non-zero exit) when it patches nothing, so a layout change cannot slip through.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

OLD = "use_auth_token"
NEW = "token"


def patch_tree(root: pathlib.Path) -> int:
    patched = 0
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if OLD in text:
            path.write_text(text.replace(OLD, NEW), encoding="utf-8")
            patched += 1
    return patched


def main() -> int:
    spec = importlib.util.find_spec("pyannote")
    if spec is None or not spec.submodule_search_locations:
        print("pyannote is not installed", file=sys.stderr)
        return 1
    total = sum(patch_tree(pathlib.Path(d)) for d in spec.submodule_search_locations)
    print(f"patched {total} pyannote files")
    if total == 0:
        print(f"no pyannote file contained {OLD}; refusing to continue", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
