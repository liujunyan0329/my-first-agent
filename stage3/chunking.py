"""阶段 3 · 文档切块：把 PDF / Markdown 变成适合检索的小片段。

RAG 的效果，七成取决于切块，三成取决于模型。这里的原则：

1. **按语义边界切**（句号、问号、分号、段落），不要按固定字数硬切；
2. **带重叠**（默认 80 字）：边界附近的答案不会因为被切断而检索不到；
3. **保留来源**（文件名 + 页码/章节）：回答时才能引用，也方便你排查检索错误；
4. **块不要太大**：500 字 ≈ 一节的核心内容；太大则一个块里混多个主题，
   embedding 被平均掉，检索反而不准。

不同切块大小的效果对比是阶段 3 的验收要求，见教程任务 ③ 的实验。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

DEFAULT_SIZE = 500
DEFAULT_OVERLAP = 80


@dataclass(frozen=True)
class Document:
    source: str            # 展示用文件名，如 "实验2_命题逻辑与SAT求解_实验任务.pdf"
    page: int | None       # PDF 页码（从 1 开始）；Markdown 为 None
    section: str           # Markdown 的最近标题；PDF 为空
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page: int | None
    section: str
    text: str

    def citation(self) -> str:
        if self.page is not None:
            return f"{self.source} p.{self.page}"
        if self.section:
            return f"{self.source} · {self.section}"
        return self.source


def normalize(text: str) -> str:
    """PDF 抽出来的文本换行混乱，统一一下，但不破坏句子边界。"""
    text = text.replace("　", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """按中英文标点和换行切句。保留标点（后面用 lookbehind，不吞字符）。"""
    parts = re.split(r"(?<=[。！？!?；;])|(?<=[.])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def chunk_text(
    text: str,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """把长文本切成 size 字左右、相邻块重叠 overlap 字的块。"""
    text = normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sentences = split_sentences(text)
    chunks: list[str] = []
    buffer = ""

    for sentence in sentences:
        # 单句超长（常见于 PDF 表格/公式），退化成硬切
        while len(sentence) > size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.append(sentence[:size])
            sentence = sentence[size - overlap:]

        if len(buffer) + len(sentence) <= size:
            buffer += sentence
        else:
            if buffer:
                chunks.append(buffer)
            buffer = (buffer[-overlap:] if overlap else "") + sentence

    if buffer.strip():
        chunks.append(buffer)

    # 过滤掉只有标点的碎片
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 30]


def _iter_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize(page.extract_text() or "")
        if len(text) >= 50:  # 跳过几乎空白/纯图片页
            documents.append(Document(path.name, page_number, "", text))
    return documents


_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")


def _iter_markdown(path: Path, label: str) -> list[Document]:
    """按标题分节，每节单独切块，块自带章节名。"""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in raw.splitlines():
        match = _HEADING.match(line)
        if match:
            sections.append((match.group(2).strip(), []))
        else:
            sections[-1][1].append(line)

    documents: list[Document] = []
    for heading, lines in sections:
        text = normalize("\n".join(lines))
        if len(text) >= 30:
            documents.append(Document(label, None, heading, text))
    return documents


def discover_sources(root: Path) -> list[Document]:
    """课程资料 = 根目录下实验任务 PDF + 各 lab 目录的指导/原理 Markdown。

    刻意排除 `实验报告.md`：那是学生自己的作业答案，混进知识库会污染检索，
    也可能被当成标准答案。
    """
    documents: list[Document] = []
    for pdf in sorted(root.glob("*.pdf")):
        documents.extend(_iter_pdf(pdf))

    keep_names = {"原理书.md", "实验指导书.md", "实验流程书.md", "README.md"}
    for lab_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for markdown in sorted(lab_dir.glob("*.md")):
            if markdown.name in keep_names:
                documents.extend(
                    _iter_markdown(markdown, f"{lab_dir.name}/{markdown.name}")
                )
    return documents


def build_chunks(
    root: Path,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in discover_sources(root):
        pieces = chunk_text(document.text, size=size, overlap=overlap)
        for index, piece in enumerate(pieces):
            page = f"p{document.page}" if document.page else "s"
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source}#{page}#{index}",
                    source=document.source,
                    page=document.page,
                    section=document.section,
                    text=piece,
                )
            )
    return chunks
