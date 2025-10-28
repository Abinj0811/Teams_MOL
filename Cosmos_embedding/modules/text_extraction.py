"""
Text Extraction Module

Full-featured text extraction for the Cosmos Embedding Admin Panel.
Based on the original text_processor_chain.py with Docling integration.

This module provides:
- Advanced PDF processing with Docling
- Authority table extraction and processing
- Committee table extraction and processing
- Multi-format document support (PDF, DOCX, TXT, MD)
- Fallback processing for reliability
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, Set, Optional, Dict, Any
import json

import pdfplumber
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    CsvFormatOption,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from tempfile import TemporaryDirectory
from pypdf import PdfReader, PdfWriter

# Import local processors
from .authority_processor import AuthorityTableProcessor
from .committee_processor import CommitteeTableProcessor

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextExtraction:
    """
    Full-featured text extraction using Docling with table processing.
    Extracts text and table content from documents into Markdown format.
    """

    def __init__(self):
        self.doc_converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.CSV,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=PyPdfiumDocumentBackend,
                ),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
                InputFormat.CSV: CsvFormatOption(pipeline_cls=SimplePipeline),
            },
        )
        self.authority = AuthorityTableProcessor()
        self.committee = CommitteeTableProcessor()

    def extract_to_markdown(self, file_path: str) -> str:
        """
        Extracts text from a file and returns Markdown output.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Extracted markdown content
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf_to_markdown(file_path)
        else:
            return self._extract_docling_any(file_path)

    def _extract_pdf_to_markdown(self, file_path: str) -> str:
        """Extract PDF content with advanced table processing."""
        logger.info(f"Extracting PDF → Markdown: {file_path}")
        page_to_parts: Dict[int, List[str]] = {}
        processed_pages: Set[int] = set()
        num_pages = 0

        # Determine number of pages and initialize map
        try:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                for i in range(1, num_pages + 1):
                    page_to_parts[i] = []
        except Exception:
            pass

        # Authority table extraction
        try:
            if self.authority.detect_authority_format(file_path):
                logger.info("Authority format detected → extracting tables")
                tables = self.authority.extract_tables(file_path)
                for t in tables:
                    p = t.get("page")
                    if p:
                        processed_pages.add(int(p))
                        # Build per-page authority markdown
                        page_num = int(p)
                        markdown = self.authority.table_to_markdown(t)
                        if markdown.strip():
                            page_to_parts[page_num].append(f"## Page {page_num} - Authority Table {t.get('table_index')}\n\n{markdown}\n")
                        # Footers/notes
                        footers = self.authority.extract_footers_and_notes(t)
                        if footers:
                            page_to_parts[page_num].append("### Notes and Additional Information\n" + "\n".join([f"- {f}" for f in footers]) + "\n")
                        # Additional page content
                        addl = self.authority.extract_page_additional_content(page_num, file_path)
                        if addl.strip():
                            page_to_parts[page_num].append("### Additional Page Content\n" + addl.strip() + "\n")
        except Exception as e:
            logger.warning(f"Authority processing error: {e}")

        # Committee table extraction
        try:
            if self.committee.detect_committee_format(file_path):
                logger.info("Committee format detected → extracting tables")
                tables = self.committee.extract_tables(file_path)
                for t in tables:
                    p = t.get("page")
                    if p:
                        processed_pages.add(int(p))
                        page_num = int(p)
                        markdown = self.committee.table_to_markdown(t)
                        if markdown.strip():
                            page_to_parts[page_num].append(f"## Page {page_num} — Committee Structure\n\n{markdown}\n")
                        addl = self.committee.extract_page_additional_content(page_num, file_path)
                        if addl.strip():
                            page_to_parts[page_num].append("### Additional Page Content\n" + addl.strip() + "\n")
        except Exception as e:
            logger.warning(f"Committee processing error: {e}")

        # Docling extraction (excluding table pages)
        try:
            logger.info("Running Docling for non-table pages...")
            docling_by_page = self._extract_docling_text_by_page(file_path, exclude_pages=processed_pages)
            for page_num, content in docling_by_page.items():
                if content.strip():
                    page_to_parts.setdefault(page_num, []).append(f"## Page {page_num}\n\n{content.strip()}\n")
        except Exception as e:
            logger.warning(f"Docling extraction failed: {e}")

        # Fallback with pdfplumber (only if we have nothing at all)
        has_any = any(part for parts in page_to_parts.values() for part in parts)
        if not has_any:
            logger.info("Docling failed → Fallback to pdfplumber")
            by_page = self._extract_pdf_with_pdfplumber_by_page(file_path)
            for page_num, content in by_page.items():
                if content.strip():
                    page_to_parts.setdefault(page_num, []).append(f"## Page {page_num}\n\n{content.strip()}\n")

        # Render pages in order
        ordered_pages = sorted(page_to_parts.keys())
        ordered_parts: List[str] = []
        for p in ordered_pages:
            ordered_parts.extend(page_to_parts[p])

        final_md = "\n\n".join([p.strip() for p in ordered_parts if p.strip()])
        return final_md.strip()

    def _extract_docling_text_by_page(self, file_path: str, exclude_pages: Optional[Set[int]] = None) -> Dict[int, str]:
        """Extract text using Docling, processing one page at a time."""
        exclude_pages = exclude_pages or set()
        by_page: Dict[int, str] = {}

        try:
            # Process one page at a time to avoid table conflicts
            reader = PdfReader(file_path)
            with TemporaryDirectory() as tmpdir:
                for idx in range(len(reader.pages)):
                    page_num = idx + 1
                    if page_num in exclude_pages:
                        continue
                    writer = PdfWriter()
                    writer.add_page(reader.pages[idx])
                    single_path = Path(tmpdir) / f"page_{page_num}.pdf"
                    with open(single_path, "wb") as f:
                        writer.write(f)

                    try:
                        conv = self.doc_converter.convert(str(single_path))
                        exported_md = None
                        # Prefer export_to_markdown when available
                        try:
                            exported_md = conv.document.export_to_markdown()
                        except Exception:
                            exported_md = None

                        if exported_md and exported_md.strip():
                            by_page[page_num] = exported_md.strip()
                            continue

                        # Fallback to elements API
                        elements = getattr(conv, "elements", [])
                        page_text_parts: List[str] = []
                        for el in elements:
                            text = getattr(el, "text", "") or getattr(el, "content", "")
                            if text and text.strip():
                                page_text_parts.append(text.strip())
                        if page_text_parts:
                            by_page[page_num] = "\n\n".join(page_text_parts)
                    except Exception as e:
                        logger.debug(f"Docling per-page conversion error on page {page_num}: {e}")
        except Exception as e:
            logger.error(f"Docling PDF extraction error: {e}")

        return by_page

    def _extract_docling_any(self, file_path: str) -> str:
        """
        Handles DOCX, TXT, CSV, JSON etc. using Docling directly.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Extracted markdown content
        """
        try:
            doc = self.doc_converter.convert(file_path)
            md_output = doc.document.export_to_markdown()
            return md_output.strip()
        except Exception as e:
            logger.error(f"Docling extraction failed for {file_path}: {e}")
            return ""

    def _extract_pdf_with_pdfplumber_by_page(self, file_path: str) -> Dict[int, str]:
        """Fallback PDF extraction using pdfplumber."""
        by_page: Dict[int, str] = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        by_page[page_num] = text.strip()
        except Exception as e:
            logger.error(f"pdfplumber extraction error: {e}")
        
        return by_page

    def extract_tables_structured(self, file_path: str) -> Dict[str, Any]:
        """
        Extract only table data (authority/committee) into a structured JSON.
        Does not include Docling text. Useful for structured data export.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing structured table data
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        result: Dict[str, Any] = {
            "source": str(path),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "tables": []  # list of {type, page, table_index, raw_data}
        }

        # Authority tables
        try:
            if self.authority.detect_authority_format(file_path):
                for t in self.authority.extract_tables(file_path):
                    result["tables"].append({
                        "type": "authority",
                        "page": t.get("page"),
                        "table_index": t.get("table_index"),
                        "raw_data": t.get("raw_data", [])
                    })
        except Exception as e:
            logger.warning(f"Authority table extraction error: {e}")

        # Committee tables
        try:
            if self.committee.detect_committee_format(file_path):
                for t in self.committee.extract_tables(file_path):
                    result["tables"].append({
                        "type": "committee",
                        "page": t.get("page"),
                        "table_index": t.get("table_index"),
                        "raw_data": t.get("raw_data", [])
                    })
        except Exception as e:
            logger.warning(f"Committee table extraction error: {e}")

        # Sort by page then table index
        result["tables"].sort(key=lambda x: (x.get("page") or 0, x.get("table_index") or 0))
        return result

    def render_tables_markdown_from_json(self, tables_structured: Dict[str, Any]) -> str:
        """
        Render table-only Markdown from a structured tables JSON.
        
        Args:
            tables_structured: Structured table data
            
        Returns:
            str: Markdown formatted table content
        """
        parts: List[str] = []
        for t in tables_structured.get("tables", []):
            ttype = t.get("type")
            page = t.get("page")
            tbl_idx = t.get("table_index")
            raw = t.get("raw_data", [])
            table_dict = {"raw_data": raw, "page": page, "table_index": tbl_idx}
            if ttype == "authority":
                md = self.authority.table_to_markdown(table_dict)
                if md.strip():
                    parts.append(f"## Page {page} - Authority Table {tbl_idx}\n\n{md}\n")
            elif ttype == "committee":
                md = self.committee.table_to_markdown(table_dict)
                if md.strip():
                    parts.append(f"## Page {page} — Committee Structure\n\n{md}\n")
        return "\n".join([p.strip() for p in parts if p.strip()]).strip()

    def render_full_markdown_from_tables_json(self, tables_structured: Dict[str, Any]) -> str:
        """
        Build full Markdown by combining table markdown from provided JSON with
        Docling text extracted from non-table pages of the original source PDF.
        
        Args:
            tables_structured: Structured table data with source path
            
        Returns:
            str: Complete markdown content
        """
        source_path = tables_structured.get("source")
        if not source_path:
            # Fall back to table-only rendering
            return self.render_tables_markdown_from_json(tables_structured)

        # Collect processed table pages and build per-page parts
        page_to_parts: Dict[int, List[str]] = {}
        processed_pages: Set[int] = set()

        for t in tables_structured.get("tables", []):
            page = int(t.get("page") or 0)
            if not page:
                continue
            processed_pages.add(page)
            page_to_parts.setdefault(page, [])
            ttype = t.get("type")
            tbl_idx = t.get("table_index")
            raw = t.get("raw_data", [])
            table_dict = {"raw_data": raw, "page": page, "table_index": tbl_idx}
            if ttype == "authority":
                md = self.authority.table_to_markdown(table_dict)
                if md.strip():
                    page_to_parts[page].append(f"## Page {page} - Authority Table {tbl_idx}\n\n{md}\n")
                # Extract and append footers/notes
                footers = self.authority.extract_footers_and_notes(table_dict)
                if footers:
                    page_to_parts[page].append("### Notes and Additional Information\n" + "\n".join([f"- {f}" for f in footers]) + "\n")
                # Append additional page content (outside tables)
                addl = self.authority.extract_page_additional_content(page, source_path)
                if addl.strip():
                    page_to_parts[page].append("### Additional Page Content\n" + addl.strip() + "\n")
            elif ttype == "committee":
                md = self.committee.table_to_markdown(table_dict)
                if md.strip():
                    page_to_parts[page].append(f"## Page {page} — Committee Structure\n\n{md}\n")
                # Append additional page content (outside tables)
                addl = self.committee.extract_page_additional_content(page, source_path)
                if addl.strip():
                    page_to_parts[page].append("### Additional Page Content\n" + addl.strip() + "\n")

        # Add Docling text for non-table pages
        try:
            docling_by_page = self._extract_docling_text_by_page(source_path, exclude_pages=processed_pages)
            for page_num, content in docling_by_page.items():
                if content and content.strip():
                    page_to_parts.setdefault(page_num, []).append(f"## Page {page_num}\n\n{content.strip()}\n")
        except Exception as e:
            logger.warning(f"Docling extraction during render failed: {e}")

        # Render in ascending page order
        ordered_pages = sorted(page_to_parts.keys())
        ordered_parts: List[str] = []
        for p in ordered_pages:
            ordered_parts.extend(page_to_parts[p])
        return "\n\n".join([p.strip() for p in ordered_parts if p.strip()]).strip()
