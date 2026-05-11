"""Path resolution for external RAG-system repos used by `rag_test`.

We assume the four external repos live as **siblings** of `rag_test/`. Default layout::

    <parent>/
    ├── HippoRAG/      ← OSU-NLP-Group/HippoRAG
    ├── raptor/        ← parthsarthi03/raptor
    ├── ComoRAG/       ← EternityJune25/ComoRAG (with patches applied — see patches/comorag)
    ├── arag/          ← Ayanami0730/arag
    └── rag_test/      ← this repo

If your clones live elsewhere, override per-system with environment variables
(`HIPPORAG_PATH`, `RAPTOR_PATH`, `COMORAG_PATH`, `ARAG_PATH`) or by copying
`config/paths.yaml.example` to `config/paths.yaml` and editing it.

Resolution priority for each system:
    1. environment variable `<SYSTEM>_PATH`
    2. `<repo_root>/config/paths.yaml` (if it exists)
    3. sibling default `<repo_root>/../<SystemDirName>`

Importable from any script under rag_test::

    from utils.paths import get_external_path
    sys.path.insert(0, get_external_path("hipporag"))
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical system -> (env var, default sibling dir name)
_SYSTEMS: Dict[str, Dict[str, str]] = {
    "hipporag": {"env": "HIPPORAG_PATH", "dir": "HippoRAG"},
    "raptor":   {"env": "RAPTOR_PATH",   "dir": "raptor"},
    "comorag":  {"env": "COMORAG_PATH",  "dir": "ComoRAG"},
    "arag":     {"env": "ARAG_PATH",     "dir": "arag"},
}

_yaml_cache: Optional[Dict[str, str]] = None


def _load_yaml_overrides() -> Dict[str, str]:
    """Load `config/paths.yaml` if present. Returns a flat dict {system: path}."""
    global _yaml_cache
    if _yaml_cache is not None:
        return _yaml_cache
    _yaml_cache = {}
    yaml_path = REPO_ROOT / "config" / "paths.yaml"
    if not yaml_path.exists():
        return _yaml_cache
    try:
        import yaml  # PyYAML
    except ImportError:  # tolerate missing yaml; just skip overrides
        return _yaml_cache
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        ext = data.get("external", {}) if isinstance(data, dict) else {}
        for k, v in ext.items():
            if isinstance(v, str) and v.strip():
                _yaml_cache[k.lower()] = v
    except Exception as e:  # noqa: BLE001
        print(f"[utils.paths] warning: failed to parse {yaml_path}: {e}")
    return _yaml_cache


def get_external_path(system: str, *, must_exist: bool = True) -> str:
    """Resolve the on-disk path to one of the four external RAG-system repos.

    Args:
        system: one of "hipporag", "raptor", "comorag", "arag" (case-insensitive).
        must_exist: if True (default), raise FileNotFoundError when the resolved
            path does not exist on disk.

    Returns:
        Absolute path as a string (suitable for `sys.path.insert(0, ...)`).
    """
    key = system.lower().strip()
    if key not in _SYSTEMS:
        raise ValueError(
            f"Unknown external system {system!r}. Valid: {sorted(_SYSTEMS)}"
        )

    meta = _SYSTEMS[key]

    # 1) env var
    env_val = os.environ.get(meta["env"], "").strip()
    if env_val:
        resolved = Path(env_val).expanduser().resolve()
    else:
        # 2) yaml override
        yaml_val = _load_yaml_overrides().get(key, "").strip()
        if yaml_val:
            # yaml values may be relative to repo root
            p = Path(yaml_val).expanduser()
            resolved = (REPO_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
        else:
            # 3) sibling default
            resolved = (REPO_ROOT.parent / meta["dir"]).resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"External repo {key!r} not found at {resolved}.\n"
            f"  - set env var {meta['env']}=<path>, OR\n"
            f"  - copy {REPO_ROOT}/config/paths.yaml.example → paths.yaml and edit, OR\n"
            f"  - git clone the repo at the sibling default ({REPO_ROOT.parent}/{meta['dir']}).\n"
            f"  See EXTERNAL_REPOS.md for clone commands and commit hashes."
        )
    return str(resolved)


def get_data_path(*segments: str) -> str:
    """Resolve a path under `rag_test/reproduce/dataset/`."""
    return str(REPO_ROOT.joinpath("reproduce", "dataset", *segments).resolve())


if __name__ == "__main__":
    print(f"rag_test root: {REPO_ROOT}")
    for sys_name in _SYSTEMS:
        try:
            p = get_external_path(sys_name, must_exist=False)
            ok = Path(p).exists()
            print(f"  {sys_name:10s} → {p}  {'[OK]' if ok else '[MISSING]'}")
        except Exception as e:  # noqa: BLE001
            print(f"  {sys_name:10s} → ERROR: {e}")
