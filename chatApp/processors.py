from pathlib import Path
from typing import Union
import docx
# import fitz  # PyMuPDF (fast PDF text extraction)
# import markdown2

def extract_text_from_file(path: Union[str, Path]) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in [".md", ".markdown", ".txt"]:
        return p.read_text(encoding="utf-8")
    if suffix in [".docx"]:
        return extract_docx(p)
    if suffix in [".pdf"]:
        return extract_pdf(p)
    raise ValueError(f"Unsupported file format: {suffix}")

def extract_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    full = []
    for para in doc.paragraphs:
        full.append(para.text)
    return "\n\n".join(full)

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus

def extract_pdf(path: Path) -> str:
    """
    Extracts text content from a PDF file using Docling.
    
    Args:
        path (Path): Path to the PDF file.
        
    Returns:
        str: Extracted text (Markdown) from the PDF.
    """
    converter = DocumentConverter()
    result = converter.convert(str(path))
    
    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"Document conversion failed with status {result.status}")
    
    # The conversion result is a rich document model.
    # Export it to markdown or plain text as required:
    text_md = result.document.export_to_markdown()
    return text_md