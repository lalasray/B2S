from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class AssetRecord:
    """Resolved asset metadata after a provider chooses a concrete file path."""

    label: str
    path: str
    source: str


class AssetProvider:
    """Common interface for local, manifest, or future online retrieval backends."""

    def resolve(self, label: str) -> AssetRecord:
        raise NotImplementedError


class LocalAssetProvider(AssetProvider):
    """Resolves assets directly from the project's local asset library."""

    def __init__(self, root: str):
        self.root = Path(root)

    def resolve(self, label: str) -> AssetRecord:
        path = self.root / f"{label}.obj"
        if not path.exists():
            raise FileNotFoundError(f"No local asset found for label {label}: {path}")
        return AssetRecord(label=label, path=str(path), source="local")


class ManifestAssetProvider(AssetProvider):
    """Resolves assets from a manifest that can point to local files or remote URLs."""

    def __init__(self, manifest: str, cache_dir: str = "Code/assets/cache"):
        self.manifest = manifest
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._records = self._load_manifest(manifest)

    def _load_manifest(self, manifest: str) -> Dict[str, Dict[str, str]]:
        # This lets the same code work with either a checked-in JSON file or a hosted manifest.
        if manifest.startswith("http://") or manifest.startswith("https://"):
            with urllib.request.urlopen(manifest) as response:
                return json.loads(response.read().decode("utf-8"))
        return json.loads(Path(manifest).read_text())

    def _download(self, url: str, suffix: str = ".obj") -> Path:
        # Cache by URL hash so repeated retrievals do not re-download the same mesh.
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        output = self.cache_dir / f"{digest}{suffix}"
        if not output.exists():
            urllib.request.urlretrieve(url, output)
        return output

    def resolve(self, label: str) -> AssetRecord:
        record = self._records.get(label)
        if record is None:
            raise KeyError(f"Label {label} not found in asset manifest")
        path = record["path"]
        # A manifest entry can either be a local path or a remote mesh URL.
        if path.startswith("http://") or path.startswith("https://"):
            suffix = Path(path).suffix or ".obj"
            cached = self._download(path, suffix=suffix)
            return AssetRecord(label=label, path=str(cached), source=path)
        return AssetRecord(label=label, path=path, source="manifest")


def make_asset_provider(mode: str, local_root: str, manifest: Optional[str] = None) -> AssetProvider:
    """Factory used by inference so callers only pass a mode string."""
    if mode == "manifest":
        if not manifest:
            raise ValueError("Manifest mode requires a manifest path or URL")
        return ManifestAssetProvider(manifest)
    return LocalAssetProvider(local_root)
