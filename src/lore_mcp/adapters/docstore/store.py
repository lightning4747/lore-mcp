"""Filesystem content-addressed DocumentStore adapter."""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from lore_mcp.adapters.docstore.errors import DocumentCorruptedError
from lore_mcp.domain.documents import NormalizedDocument, RawDocument
from lore_mcp.domain.types import SeriesId
from lore_mcp.ports.documents import DocumentStore

logger = logging.getLogger(__name__)


def compute_sha256(data: str) -> str:
    """Compute hex SHA-256 digest of UTF-8 string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class FilesystemDocumentStore(DocumentStore):
    """Content-addressed store persisting raw and normalized documents on disk."""

    def __init__(self, base_dir: Path | str = Path("data/store")) -> None:
        self.base_dir: Path = Path(base_dir)

    def _raw_objects_dir(self, series_id: SeriesId | str) -> Path:
        return self.base_dir / str(series_id) / "raw" / "objects"

    def _raw_urls_dir(self, series_id: SeriesId | str) -> Path:
        return self.base_dir / str(series_id) / "raw" / "by_url"

    def _norm_objects_dir(self, series_id: SeriesId | str) -> Path:
        return self.base_dir / str(series_id) / "normalized" / "objects"

    def _norm_urls_dir(self, series_id: SeriesId | str) -> Path:
        return self.base_dir / str(series_id) / "normalized" / "by_url"

    def _atomic_write(self, target_path: Path, content: str) -> None:
        """Atomically write text to target_path using a temporary sibling file."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.parent / f".tmp.{uuid.uuid4().hex}"
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    async def put_raw(self, doc: RawDocument) -> None:
        """Store a raw document content-addressed and update URL index."""
        content_hash = compute_sha256(doc.content)
        obj_file = self._raw_objects_dir(doc.series_id) / f"{content_hash}.json"

        # 1. Write content-addressed payload
        payload = doc.model_dump_json(indent=2)
        await asyncio.to_thread(self._atomic_write, obj_file, payload)

        # 2. Write URL pointer index
        url_hash = compute_sha256(doc.source_url)
        url_file = self._raw_urls_dir(doc.series_id) / f"{url_hash}.json"
        pointer = json.dumps(
            {
                "source_url": doc.source_url,
                "content_hash": content_hash,
            },
            indent=2,
        )
        await asyncio.to_thread(self._atomic_write, url_file, pointer)

    async def get_raw(
        self,
        series_id: SeriesId,
        source_url: str,
    ) -> RawDocument | None:
        """Retrieve a raw document by series and source URL."""
        url_hash = compute_sha256(source_url)
        url_file = self._raw_urls_dir(series_id) / f"{url_hash}.json"

        if not await asyncio.to_thread(url_file.is_file):
            return None

        pointer_text = await asyncio.to_thread(url_file.read_text, encoding="utf-8")
        try:
            pointer_data = json.loads(pointer_text)
            content_hash = str(pointer_data["content_hash"])
        except (json.JSONDecodeError, KeyError):
            return None

        obj_file = self._raw_objects_dir(series_id) / f"{content_hash}.json"
        if not await asyncio.to_thread(obj_file.is_file):
            return None

        doc_text = await asyncio.to_thread(obj_file.read_text, encoding="utf-8")
        return RawDocument.model_validate_json(doc_text)

    async def put_normalized(self, doc: NormalizedDocument) -> None:
        """Store a normalized document content-addressed and update URL index."""
        content_hash = doc.content_hash or compute_sha256(doc.text)
        obj_file = self._norm_objects_dir(doc.series_id) / f"{content_hash}.json"

        # 1. Write content-addressed payload
        payload = doc.model_dump_json(indent=2)
        await asyncio.to_thread(self._atomic_write, obj_file, payload)

        # 2. Write URL pointer index
        url_hash = compute_sha256(doc.source_url)
        url_file = self._norm_urls_dir(doc.series_id) / f"{url_hash}.json"
        pointer = json.dumps(
            {
                "source_url": doc.source_url,
                "content_hash": content_hash,
            },
            indent=2,
        )
        await asyncio.to_thread(self._atomic_write, url_file, pointer)

    async def get_normalized(
        self,
        series_id: SeriesId,
        source_url: str,
    ) -> NormalizedDocument | None:
        """Retrieve a normalized document by series and source URL."""
        url_hash = compute_sha256(source_url)
        url_file = self._norm_urls_dir(series_id) / f"{url_hash}.json"

        if not await asyncio.to_thread(url_file.is_file):
            return None

        pointer_text = await asyncio.to_thread(url_file.read_text, encoding="utf-8")
        try:
            pointer_data = json.loads(pointer_text)
            content_hash = str(pointer_data["content_hash"])
        except (json.JSONDecodeError, KeyError):
            return None

        obj_file = self._norm_objects_dir(series_id) / f"{content_hash}.json"
        if not await asyncio.to_thread(obj_file.is_file):
            return None

        doc_text = await asyncio.to_thread(obj_file.read_text, encoding="utf-8")
        return NormalizedDocument.model_validate_json(doc_text)

    async def list_urls(self, series_id: SeriesId) -> list[str]:
        """List all stored document URLs for a series across raw and normalized."""
        urls: set[str] = set()

        for u_dir in (self._raw_urls_dir(series_id), self._norm_urls_dir(series_id)):
            if not await asyncio.to_thread(u_dir.is_dir):
                continue
            pointer_files = await asyncio.to_thread(
                lambda d=u_dir: list(d.glob("*.json"))
            )
            for p_file in pointer_files:
                try:
                    text = await asyncio.to_thread(p_file.read_text, encoding="utf-8")
                    data = json.loads(text)
                    if "source_url" in data:
                        urls.add(str(data["source_url"]))
                except Exception:
                    continue

        return sorted(urls)

    async def list_series(self) -> list[SeriesId]:
        """List all series identifiers currently stored."""
        if not await asyncio.to_thread(self.base_dir.is_dir):
            return []

        series: list[SeriesId] = []
        entries = await asyncio.to_thread(lambda: list(self.base_dir.iterdir()))
        for entry in sorted(entries):
            if entry.is_dir() and not entry.name.startswith("."):
                raw_or_norm = (entry / "raw").is_dir() or (
                    entry / "normalized"
                ).is_dir()
                if raw_or_norm:
                    series.append(SeriesId(entry.name))
        return series

    async def iter_raw(self, series_id: SeriesId) -> AsyncIterator[RawDocument]:
        """Iterate over all raw documents stored for a series."""
        objs_dir = self._raw_objects_dir(series_id)
        if not await asyncio.to_thread(objs_dir.is_dir):
            return

        obj_files = await asyncio.to_thread(lambda: sorted(objs_dir.glob("*.json")))
        for obj_file in obj_files:
            text = await asyncio.to_thread(obj_file.read_text, encoding="utf-8")
            doc = RawDocument.model_validate_json(text)
            # Verify content-address integrity
            actual_hash = compute_sha256(doc.content)
            expected_hash = obj_file.stem
            if actual_hash != expected_hash:
                raise DocumentCorruptedError(
                    path=str(obj_file),
                    expected_hash=expected_hash,
                    actual_hash=actual_hash,
                )
            yield doc

    async def iter_normalized(
        self,
        series_id: SeriesId,
    ) -> AsyncIterator[NormalizedDocument]:
        """Iterate over all normalized documents stored for a series."""
        objs_dir = self._norm_objects_dir(series_id)
        if not await asyncio.to_thread(objs_dir.is_dir):
            return

        obj_files = await asyncio.to_thread(lambda: sorted(objs_dir.glob("*.json")))
        for obj_file in obj_files:
            text = await asyncio.to_thread(obj_file.read_text, encoding="utf-8")
            doc = NormalizedDocument.model_validate_json(text)
            actual_hash = doc.content_hash or compute_sha256(doc.text)
            expected_hash = obj_file.stem
            if actual_hash != expected_hash:
                raise DocumentCorruptedError(
                    path=str(obj_file),
                    expected_hash=expected_hash,
                    actual_hash=actual_hash,
                )
            yield doc

    async def rebuild_url_index(
        self,
        series_id: SeriesId | None = None,
    ) -> int:
        """Rebuild URL indexes from stored objects without network access [S6]."""
        target_series = [series_id] if series_id else await self.list_series()
        rebuilt_count = 0

        for sid in target_series:
            # Rebuild raw URL pointers
            raw_objs_dir = self._raw_objects_dir(sid)
            if await asyncio.to_thread(raw_objs_dir.is_dir):
                raw_files = await asyncio.to_thread(
                    lambda d=raw_objs_dir: list(d.glob("*.json"))
                )
                for f in raw_files:
                    text = await asyncio.to_thread(f.read_text, encoding="utf-8")
                    doc = RawDocument.model_validate_json(text)
                    content_hash = compute_sha256(doc.content)
                    url_hash = compute_sha256(doc.source_url)
                    url_file = self._raw_urls_dir(sid) / f"{url_hash}.json"
                    pointer = json.dumps(
                        {
                            "source_url": doc.source_url,
                            "content_hash": content_hash,
                        },
                        indent=2,
                    )
                    await asyncio.to_thread(self._atomic_write, url_file, pointer)
                    rebuilt_count += 1

            # Rebuild normalized URL pointers
            norm_objs_dir = self._norm_objects_dir(sid)
            if await asyncio.to_thread(norm_objs_dir.is_dir):
                norm_files = await asyncio.to_thread(
                    lambda d=norm_objs_dir: list(d.glob("*.json"))
                )
                for f in norm_files:
                    text = await asyncio.to_thread(f.read_text, encoding="utf-8")
                    norm_doc = NormalizedDocument.model_validate_json(text)
                    content_hash = norm_doc.content_hash or compute_sha256(
                        norm_doc.text
                    )
                    url_hash = compute_sha256(norm_doc.source_url)
                    url_file = self._norm_urls_dir(sid) / f"{url_hash}.json"
                    pointer = json.dumps(
                        {
                            "source_url": norm_doc.source_url,
                            "content_hash": content_hash,
                        },
                        indent=2,
                    )
                    await asyncio.to_thread(self._atomic_write, url_file, pointer)
                    rebuilt_count += 1

        return rebuilt_count
