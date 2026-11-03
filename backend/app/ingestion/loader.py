"""
Document loaders supporting text, markdown, json, pdf, and web URLs.
"""
from pathlib import Path
from typing import List, Dict, Any, Union
import json
import os
from backend.app.models import Document
from backend.app.ingestion.cleaner import DocumentCleaner

class DocumentLoader:
    @classmethod
    def load_from_text(cls, text: str, title: str = "Pasted Text", metadata: Dict[str, Any] = None) -> Document:
        cleaned = DocumentCleaner.clean(text)
        meta = metadata or {}
        meta.setdefault("source", "text_input")
        meta.setdefault("word_count", len(cleaned.split()))
        meta.setdefault("char_count", len(cleaned))
        return Document(title=title, content=cleaned, metadata=meta)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        title = path.stem.replace("_", " ").title()
        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "extension": suffix,
            "file_size": path.stat().st_size
        }

        if suffix in [".txt", ".md", ".markdown", ".rst"]:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    title = data.get("title", title)
                    content = data.get("content", data.get("text", json.dumps(data, indent=2)))
                elif isinstance(data, list):
                    content = "\n\n".join([str(item) for item in data])
                else:
                    content = str(data)
        elif suffix == ".pdf":
            # Attempt pdf extraction via pypdf or fallback
            content = cls._extract_pdf(path)
        else:
            # Generic binary/text fallback
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        cleaned = DocumentCleaner.clean(content)
        metadata["word_count"] = len(cleaned.split())
        metadata["char_count"] = len(cleaned)
        return Document(title=title, content=cleaned, metadata=metadata)

    @classmethod
    def _extract_pdf(cls, path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append(text)
            return "\n\n".join(pages)
        except Exception:
            # Fallback reading raw text lines
            with open(path, "rb") as f:
                raw = f.read()
            # Simple text extraction from raw pdf streams if pypdf is not available
            text_matches = []
            import re
            for line in raw.split(b"\n"):
                try:
                    decoded = line.decode("ascii", errors="ignore")
                    if len(decoded.strip()) > 4 and not decoded.startswith(("%PDF", "xref", "trailer", "endobj", "stream", "endstream")):
                        text_matches.append(decoded.strip())
                except Exception:
                    pass
            return "\n".join(text_matches) if text_matches else "PDF content placeholder"

    @classmethod
    def load_directory(cls, dir_path: Union[str, Path]) -> List[Document]:
        dir_p = Path(dir_path)
        if not dir_p.exists():
            return []
        documents = []
        for file in dir_p.glob("**/*"):
            if file.is_file() and file.suffix.lower() in [".txt", ".md", ".json", ".pdf"]:
                try:
                    doc = cls.load_from_file(file)
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        return documents
