"""
Text cleaner and normalizer for ingestion pipeline.
"""
import re
import unicodedata

class DocumentCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        # Replace non-breaking spaces and tabs with normal space
        text = text.replace("\u00a0", " ").replace("\t", " ")
        # Standardize newlines
        text = re.sub(r"\r\n|\r", "\n", text)
        # Collapse excessive newlines (more than 2 consecutive newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces into single space on the same line
        text = re.sub(r"[ ]{2,}", " ", text)
        # Strip trailing/leading spaces on lines
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()
