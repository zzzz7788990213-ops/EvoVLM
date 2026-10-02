"""Fail-closed structural, textual, and rendered-page QA for the final paper PDF."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Sequence

import fitz

REQUIRED_TEXT = (
    "Formal Recovery-v2 Evaluation",
    "Whole-set Speed32 replacement",
    "Negative Results and Failure Analysis",
    "Reproducibility and Artifact Traceability",
    "evolved_instruct",
)


class PdfQaError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> Path:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode) or path.stat().st_size <= 0:
        raise PdfQaError(f"not a nonempty regular file: {path}")
    return path


def write_json_exclusive(path: Path, document: dict[str, object]) -> None:
    payload = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def run_qa(
    *, pdf: Path, tex: Path, entry: Path, style: Path, log: Path, render_dir: Path, receipt: Path,
    compiler: str,
) -> dict[str, object]:
    for path in (pdf, tex, entry, style, log):
        regular(path)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    if "neurips_2025.sty" not in log_text:
        raise PdfQaError("compiler log does not prove neurips_2025.sty was loaded")
    payload = pdf.read_bytes()
    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-2048:]:
        raise PdfQaError("PDF framing is invalid")
    if render_dir.exists() or render_dir.is_symlink():
        raise PdfQaError("render directory already exists")
    render_dir.mkdir(mode=0o700)

    document = fitz.open(pdf)
    try:
        page_count = document.page_count
        if not 6 <= page_count <= 8:
            raise PdfQaError(f"page count {page_count} is outside 6--8")
        metadata = dict(document.metadata or {})
        full_text_parts: list[str] = []
        pages: list[dict[str, object]] = []
        for index, page in enumerate(document):
            rect = page.rect
            if not (500.0 <= rect.width <= 700.0 and 700.0 <= rect.height <= 900.0):
                raise PdfQaError(f"page {index + 1} dimensions are implausible")
            text = page.get_text("text")
            full_text_parts.append(text)
            if len(text.strip()) < 200:
                raise PdfQaError(f"page {index + 1} has too little extractable text")
            blocks = page.get_text("blocks")
            if not blocks:
                raise PdfQaError(f"page {index + 1} has no text blocks")
            for block in blocks:
                x0, y0, x1, y1 = (float(block[i]) for i in range(4))
                if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
                    raise PdfQaError(f"page {index + 1} has non-finite block geometry")
                if x0 < -0.5 or y0 < -0.5 or x1 > rect.width + 0.5 or y1 > rect.height + 0.5:
                    raise PdfQaError(f"page {index + 1} has clipped text geometry")
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            png = render_dir / f"page-{index + 1:02d}.png"
            pix.save(png)
            regular(png)
            samples = memoryview(pix.samples)
            channels = pix.n
            nonwhite = 0
            for offset in range(0, len(samples), channels):
                if min(samples[offset:offset + min(3, channels)]) < 245:
                    nonwhite += 1
            fraction = nonwhite / (pix.width * pix.height)
            if not 0.01 <= fraction <= 0.65:
                raise PdfQaError(
                    f"page {index + 1} rendered ink fraction {fraction:.6f} is implausible"
                )
            pages.append(
                {
                    "page": index + 1,
                    "width_points": rect.width,
                    "height_points": rect.height,
                    "text_characters": len(text),
                    "text_block_count": len(blocks),
                    "render_width": pix.width,
                    "render_height": pix.height,
                    "render_nonwhite_fraction": fraction,
                    "render_sha256": sha256_file(png),
                    "render_size_bytes": png.stat().st_size,
                }
            )
    finally:
        document.close()
    full_text = "\n".join(full_text_parts)
    missing = [phrase for phrase in REQUIRED_TEXT if phrase not in full_text]
    if missing:
        raise PdfQaError(f"required PDF text is missing: {missing}")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "official_neurips_style_loaded": True,
        "compiler": compiler,
        "pdf": {
            "sha256": sha256_file(pdf),
            "size_bytes": pdf.stat().st_size,
            "page_count": page_count,
            "metadata": metadata,
        },
        "sources": {
            "main_tex_sha256": sha256_file(tex),
            "tectonic_entry_sha256": sha256_file(entry),
            "neurips_2025_style_sha256": sha256_file(style),
            "compiler_log_sha256": sha256_file(log),
        },
        "required_text": {phrase: True for phrase in REQUIRED_TEXT},
        "pages": pages,
    }
    write_json_exclusive(receipt, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    args = parser.parse_args(argv)
    try:
        report = run_qa(
            pdf=args.pdf, tex=args.tex, entry=args.entry, style=args.style, log=args.log,
            render_dir=args.render_dir, receipt=args.receipt, compiler=args.compiler,
        )
    except (OSError, PdfQaError, RuntimeError, ValueError) as exc:
        print(f"PDF QA failed: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
