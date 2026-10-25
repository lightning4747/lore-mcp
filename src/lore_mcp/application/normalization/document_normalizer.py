"""Normalizer for general wiki, fandom, and HTML source documents (Task 3.1)."""

import hashlib
import html
import re
import unicodedata

from bs4 import BeautifulSoup

from lore_mcp.domain.documents import DocumentSection, NormalizedDocument, RawDocument
from lore_mcp.domain.normalization import DocumentNormalizationOptions
from lore_mcp.ports.normalization import DocumentNormalizer

# Elements to strip completely from HTML DOM
UNWANTED_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "nav",
    "footer",
    "ref",
}
UNWANTED_CLASSES = {
    "navbox",
    "toc",
    "mw-jump-link",
    "catlinks",
    "reflist",
    "printfooter",
    "mw-editsection",
    "noprint",
    "ambox",
    "hatnote",
    "navigation-box",
}
INFOBOX_CLASSES = {"infobox", "portable-infobox"}

# Wikitext patterns
WIKITEXT_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
WIKITEXT_REF = re.compile(
    r"<ref(?:\s+[^>]*?)?>.*?</ref>|<ref(?:\s+[^>]*?)?/>",
    re.DOTALL | re.IGNORECASE,
)
WIKITEXT_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
WIKITEXT_MAGIC = re.compile(r"__[A-Z]+__")
WIKITEXT_WIKILINK_PIPED = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
WIKITEXT_EXTLINK = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
WIKITEXT_BOLD_ITALIC = re.compile(r"'''''?([^']+)'''''?")
WIKITEXT_HEADING = re.compile(r"^[ \t]*(={1,6})\s*(.*?)\s*\1[ \t]*$", re.MULTILINE)

# Markdown heading line pattern
MARKDOWN_HEADING = re.compile(r"^[ \t]*(#{1,6})\s+(.+)$")

# Redirect pattern
RAW_REDIRECT_PATTERN = re.compile(
    r"^(?:#REDIRECT|REDIRECT)\s*(?:\[\[\s*)?([^\]\r\n]+?)(?:\s*\]\])?$",
    re.IGNORECASE,
)


def compute_sha256(text: str) -> str:
    """Compute deterministic SHA-256 hex digest of string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DefaultDocumentNormalizer(DocumentNormalizer):
    """Normalizes raw wiki, HTML, and text documents into NormalizedDocument."""

    def __init__(
        self, default_options: DocumentNormalizationOptions | None = None
    ) -> None:
        self._default_options = default_options or DocumentNormalizationOptions()

    def _strip_html(self, raw_html: str, options: DocumentNormalizationOptions) -> str:
        """Parse HTML, decompose noise elements, convert headings, and extract text."""
        soup = BeautifulSoup(raw_html, "html.parser")

        # 1. Remove unwanted tags
        for tag in soup.find_all(UNWANTED_TAGS):
            tag.decompose()

        # 2. Remove unwanted class-based containers
        for class_name in UNWANTED_CLASSES:
            for elem in soup.find_all(
                class_=lambda c, cn=class_name: bool(c and cn in c.split())
            ):
                elem.decompose()

        # 3. Strip infoboxes if requested
        if options.strip_infoboxes:
            for infobox_class in INFOBOX_CLASSES:
                for elem in soup.find_all(
                    class_=lambda c, ic=infobox_class: bool(c and ic in c.split())
                ):
                    elem.decompose()

        # 4. Convert headings h1-h6 into Markdown headings
        for i in range(1, 7):
            for h in soup.find_all(f"h{i}"):
                heading_text = h.get_text(strip=True)
                if heading_text:
                    h.replace_with(f"\n\n{'#' * i} {heading_text}\n\n")
                else:
                    h.decompose()

        # 5. Convert paragraphs and line breaks
        for p in soup.find_all("p"):
            p_text = p.get_text(strip=True)
            if p_text:
                p.replace_with(f"\n\n{p_text}\n\n")
            else:
                p.decompose()

        for br in soup.find_all("br"):
            br.replace_with("\n")

        for li in soup.find_all("li"):
            li_text = li.get_text(strip=True)
            if li_text:
                li.replace_with(f"\n* {li_text}\n")
            else:
                li.decompose()

        return soup.get_text()

    def _clean_wikitext(self, text: str) -> str:
        """Strip Wikitext templates, links, comments, and markup."""
        # 1. Comments and references
        cleaned = WIKITEXT_COMMENT.sub("", text)
        cleaned = WIKITEXT_REF.sub("", cleaned)

        # 2. Template macros (repeating passes for nested templates)
        for _ in range(3):
            cleaned = WIKITEXT_TEMPLATE.sub("", cleaned)

        # 3. Magic words
        cleaned = WIKITEXT_MAGIC.sub("", cleaned)

        # 4. Links
        cleaned = WIKITEXT_WIKILINK_PIPED.sub(r"\1", cleaned)
        cleaned = WIKITEXT_EXTLINK.sub(r"\1", cleaned)

        # 5. Bold / italics
        cleaned = WIKITEXT_BOLD_ITALIC.sub(r"\1", cleaned)

        # 6. Wikitext headings == Heading == -> ## Heading
        def _sub_heading(match: re.Match[str]) -> str:
            level = len(match.group(1))
            title = match.group(2).strip()
            return f"\n\n{'#' * level} {title}\n\n"

        cleaned = WIKITEXT_HEADING.sub(_sub_heading, cleaned)
        return cleaned

    def _normalize_text(
        self, text: str, options: DocumentNormalizationOptions
    ) -> str:
        """Normalize Unicode NFC, characters, and paragraph whitespace."""
        # Unescape HTML entities
        result = html.unescape(text)

        # Unicode NFC normalization
        if options.nfc_normalize:
            result = unicodedata.normalize("NFC", result)

        # Replace non-breaking spaces and zero-width spaces
        result = (
            result.replace("\u00a0", " ")
            .replace("\u200b", "")
            .replace("\ufeff", "")
        )

        # Normalize line endings
        result = result.replace("\r\n", "\n").replace("\r", "\n")

        # Clean line by line
        lines = [line.strip() for line in result.split("\n")]

        # Collapse multiple empty lines into double newline paragraphs (\n\n)
        cleaned_paragraphs: list[str] = []
        current_p_lines: list[str] = []

        for line in lines:
            if not line:
                if current_p_lines:
                    cleaned_paragraphs.append(" ".join(current_p_lines))
                    current_p_lines = []
            elif line.startswith("#") or line.startswith("*"):
                # Headings or bullet points stay on dedicated lines
                if current_p_lines:
                    cleaned_paragraphs.append(" ".join(current_p_lines))
                    current_p_lines = []
                cleaned_paragraphs.append(line)
            else:
                # Collapse excessive internal horizontal whitespace
                collapsed_line = re.sub(r"[ \t]+", " ", line)
                current_p_lines.append(collapsed_line)

        if current_p_lines:
            cleaned_paragraphs.append(" ".join(current_p_lines))

        return "\n\n".join(cleaned_paragraphs)

    def _extract_sections(
        self, full_text: str, default_title: str
    ) -> list[DocumentSection]:
        """Parse text into DocumentSection objects partitioned by markdown headings."""
        lines = full_text.split("\n")
        sections: list[DocumentSection] = []

        current_heading = default_title or "Overview"
        current_lines: list[str] = []

        for line in lines:
            heading_match = MARKDOWN_HEADING.match(line)
            if heading_match:
                if current_lines:
                    content_str = "\n".join(current_lines).strip()
                    if content_str:
                        sections.append(
                            DocumentSection(
                                heading=current_heading,
                                content=content_str,
                            )
                        )
                    current_lines = []
                current_heading = heading_match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            content_str = "\n".join(current_lines).strip()
            if content_str:
                sections.append(
                    DocumentSection(
                        heading=current_heading,
                        content=content_str,
                    )
                )

        if not sections and full_text.strip():
            sections.append(
                DocumentSection(
                    heading=default_title or "Overview",
                    content=full_text.strip(),
                )
            )

        return sections

    def normalize(
        self,
        doc: RawDocument,
        options: DocumentNormalizationOptions | None = None,
    ) -> NormalizedDocument:
        """Clean raw source text, strip noise, and extract sections."""
        opts = options or self._default_options
        content = doc.content

        # 1. Wikitext cleaning (removes <ref>, comments, templates, headings)
        cleaned_wiki = self._clean_wikitext(content)

        # 2. HTML stripping if content contains HTML
        if "<" in cleaned_wiki and ">" in cleaned_wiki:
            cleaned_html = self._strip_html(cleaned_wiki, opts)
        else:
            cleaned_html = cleaned_wiki

        # 3. Unicode and paragraph normalization
        normalized_text = self._normalize_text(cleaned_html, opts)

        # 4. Extract title from metadata or first heading
        title = doc.metadata.get("title") or ""
        if not title:
            first_line = normalized_text.split("\n", 1)[0]
            if first_line.startswith("#"):
                title = first_line.lstrip("#").strip()
            else:
                title = "Untitled Document"

        # 5. Extract heading sections
        sections = (
            self._extract_sections(normalized_text, title)
            if opts.preserve_headings
            else [DocumentSection(heading=title, content=normalized_text)]
        )

        # 6. Compute deterministic content hash
        content_hash = compute_sha256(normalized_text)

        # 7. Preserve metadata attributes and detect redirects
        cleaned_meta: dict[str, str] = {
            k: v for k, v in doc.metadata.items() if v
        }
        cleaned_meta["content_hash"] = content_hash
        cleaned_meta["title"] = title

        redirect_match = (
            RAW_REDIRECT_PATTERN.match(doc.content.strip())
            or RAW_REDIRECT_PATTERN.match(normalized_text.strip())
        )
        if redirect_match:
            cleaned_meta["redirect_target"] = redirect_match.group(1).strip()

        return NormalizedDocument(
            series_id=doc.series_id,
            source_url=doc.source_url,
            source_type=doc.source_type,
            title=title,
            text=normalized_text,
            sections=sections,
            content_hash=content_hash,
            retrieval_timestamp=doc.retrieval_timestamp,
            metadata=cleaned_meta,
        )
