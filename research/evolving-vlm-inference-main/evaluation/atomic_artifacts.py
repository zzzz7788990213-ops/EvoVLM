"""Evo-confined, durable, atomic no-replace artifact commits.

The final publication operation is a same-directory hard-link.  link(2) is
atomic and fails with EEXIST when another writer wins; it never replaces the
winner.  The temporary inode and containing directory are fsynced explicitly.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, BinaryIO, TextIO
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent

def _default_runtime_root() -> Path:
    if PROJECT_ROOT.parent.name == "workspace":
        return PROJECT_ROOT.parents[1]
    return PROJECT_ROOT

EVO_ROOT = Path(
    os.environ.get("EVOLVING_VLM_RUNTIME_ROOT", str(_default_runtime_root()))
).expanduser().resolve(strict=False)


def _parent_dirfd(path: Path) -> tuple[Path, int]:
    if not path.is_absolute():
        raise ValueError("artifact path must be absolute")
    if ".." in path.parts:
        raise ValueError("artifact path must not contain parent traversal")
    if path.name in {"", ".", ".."} or path.parent == path:
        raise ValueError("invalid artifact basename")
    parent = path.parent.resolve(strict=True)
    if not parent.is_relative_to(EVO_ROOT):
        raise ValueError(f"artifact parent escaped Evo: {parent}")
    relative = parent.relative_to(EVO_ROOT)
    current = EVO_ROOT
    root_meta = os.lstat(current)
    if stat.S_ISLNK(root_meta.st_mode) or not stat.S_ISDIR(root_meta.st_mode):
        raise ValueError("Evo root is not a physical directory")
    for part in relative.parts:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"artifact parent chain is unsafe: {current}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return parent, os.open(parent, flags)


def fsync_parent(path: Path) -> None:
    _, dirfd = _parent_dirfd(path)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def _publish_temp(dirfd: int, temporary: str, final: str) -> None:
    try:
        os.link(
            temporary,
            final,
            src_dir_fd=dirfd,
            dst_dir_fd=dirfd,
            follow_symlinks=False,
        )
    except FileExistsError:
        raise FileExistsError(f"refusing to replace existing artifact: {final}") from None
    os.fsync(dirfd)
    os.unlink(temporary, dir_fd=dirfd)
    os.fsync(dirfd)


def write_bytes_noreplace(path: Path, payload: bytes, mode: int = 0o600) -> None:
    _, dirfd = _parent_dirfd(path)
    temporary = f".{path.name}.{uuid4().hex}.partial"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=dirfd,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_temp(dirfd, temporary, path.name)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=dirfd)
            os.fsync(dirfd)
        except FileNotFoundError:
            pass
        os.close(dirfd)


def write_json_noreplace(path: Path, document: Any) -> None:
    payload = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    write_bytes_noreplace(path, payload)


def commit_existing_noreplace(temporary: Path, final: Path) -> None:
    if temporary.parent != final.parent:
        raise ValueError("temporary and final artifacts must share a directory")
    _, dirfd = _parent_dirfd(final)
    try:
        temp_meta = os.stat(temporary.name, dir_fd=dirfd, follow_symlinks=False)
        if stat.S_ISLNK(temp_meta.st_mode) or not stat.S_ISREG(temp_meta.st_mode):
            raise ValueError("temporary artifact is not a regular file")
        descriptor = os.open(
            temporary.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=dirfd,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _publish_temp(dirfd, temporary.name, final.name)
    finally:
        os.close(dirfd)


def open_binary_exclusive(path: Path) -> BinaryIO:
    _, dirfd = _parent_dirfd(path)
    try:
        descriptor = os.open(
            path.name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=dirfd,
        )
        os.fsync(dirfd)
    finally:
        os.close(dirfd)
    return os.fdopen(descriptor, "wb")


def open_text_exclusive(path: Path) -> TextIO:
    binary = open_binary_exclusive(path)
    return __import__("io").TextIOWrapper(binary, encoding="utf-8")


def finalize_stream(stream: BinaryIO | TextIO, path: Path) -> None:
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()
    fsync_parent(path)
