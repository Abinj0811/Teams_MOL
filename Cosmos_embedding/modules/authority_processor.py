"""
Authority Table Processor

Handles extraction and processing of authority approval tables from PDFs.
Full-featured version based on the original authority_processor.py.

This module processes specialized authority table formats that contain:
- Classification hierarchies
- Approval workflows
- Authority levels (MOL, BDM, A1-A5, etc.)
- Co-management and deliberation information
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber
from loguru import logger

logger = logging.getLogger(__name__)


class AuthorityTableProcessor:
    """Integrated processor for authority table format PDFs."""
    
    def __init__(self):
        # Unique bullet characters mapped to levels (2 = second-level, 3 = third-level)
        self.bullet_levels = {
            "♦": 2, "◆": 2, "◇": 2, "◊": 2, "▪": 2, "▫": 2, "": 2, "◻": 2, "□": 2,
            "•": 3, "●": 3, "■": 3, "◾": 3, "▸": 3, "▶": 3, "◦": 3
        }
        # build a safe regex character class including hyphens/dashes and common bullet glyphs
        escaped_chars = re.escape("".join(self.bullet_levels.keys()))
        dash_chars = r"\-\–\—"
        self.bullet_prefix_re = re.compile(rf"^\s*([{escaped_chars}{dash_chars}])\s*")

    
    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text.strip())
        return text.lower()
    
    def clean_cell(self, cell):
        return str(cell).strip().replace("\n", " ") if cell else ""
    
    def detect_authority_format(self, file_path: str) -> bool:
        """
        Detect if a PDF contains authority tables.
        
        Uses keyword matching in table headers to identify authority format.
        Scans first 10 pages for efficiency.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            True if authority format detected, False otherwise
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                authority_keywords = [
                    's/n', 'classification', 'authorised', 'approver', 'approval',
                    'co-mgmt', 'deliberation', 'report', 'review', 'cc',
                    'mol', 'bdm', 'gpm', 'authority', 'management'
                ]
                
                for page_num, page in enumerate(pdf.pages[:10], start=1):
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables, start=1):
                        if not table or len(table) < 2:
                            continue
                        header_text = ""
                        for row_idx in range(min(2, len(table))):
                            row = table[row_idx]
                            for cell in row:
                                if cell:
                                    header_text += " " + self.normalize_text(str(cell))
                        header_text = header_text.lower()
                        keyword_count = sum(1 for keyword in authority_keywords 
                                          if keyword in header_text)
                        if keyword_count >= 3:
                            logger.info(f"Authority table detected on page {page_num}, table {t_idx}")
                            return True
        except Exception as e:
            logger.error(f"Error detecting authority format: {e}")
        return False
    
    def is_header_row(self, row):
        """Check if a row is a header row for authority tables."""
        # normalize and join cells, require presence of 's/n' and 'classification'
        norm_row = [self.normalize_text(c).replace(" ", "") for c in row if c]
        joined = " ".join(norm_row)
        if "s/n" not in joined or "classification" not in joined:
            return False
        if len(row) < 6:
            return False
        return True
    
    def extract_tables(self, file_path: str, save_dir: Optional[Path] = None) -> List[Dict]:
        """
        Extract authority tables from PDF.
        
        Args:
            file_path: Path to the PDF file
            save_dir: Optional directory to save extracted data (not used for embeddings)
            
        Returns:
            List of dictionaries containing table data with metadata
        """
        all_docs = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables, start=1):
                    clean_table = [[self.clean_cell(c) for c in row] for row in table if any(row)]
                    if not clean_table:
                        continue
                    if not self.is_header_row(clean_table[0]):
                        continue
                    doc = {
                        "id": f"authority_p{page_num}_t{t_idx}",
                        "page": page_num,
                        "table_index": t_idx,
                        "raw_data": clean_table
                    }
                    all_docs.append(doc)
        return all_docs
    
    def _strip_bullet(self, text):
        """Return (level, cleaned_text). If no known bullet, level=None and text stripped."""
        if not text:
            return None, ""
        t = text.replace("\xa0", " ").strip()
        m = self.bullet_prefix_re.match(t)
        if m:
            bullet = m.group(1)
            level = self.bullet_levels.get(bullet, None)
            cleaned = t[m.end():].strip()
            return level, cleaned
        return None, t
    
    def clean_marker(self, cell_value: str) -> bool:
        """Returns True if the cell contains a ● or • marker (with or without numbers)."""
        if not cell_value:
            return False
        marker = cell_value.strip()
        return (marker.startswith("●") or marker.startswith("•") or 
                re.match(r'^[●•]\s*\(\d+\)', marker) is not None)
    
    def is_numbered_marker(self, cell_value: str) -> bool:
        """Returns True if the cell contains a numbered marker like ●(28) or ● (28)."""
        if not cell_value:
            return False
        marker = cell_value.strip()
        return re.match(r'^[●•]\s*\(\d+\)', marker) is not None
    
    def categorize_approver(self, header, header_row_1_value):
        """Categorize approver based on header and top-level header value."""
        header = (header or "").strip()
        top = (header_row_1_value or "").strip().lower()
        
        if "co-mgmt" in top or "co-management" in top:
            return "Co-Management Dept."
        elif "deliberation" in top:
            return "Deliberation"
        elif "report" in top:
            return "Report"
        elif "review" in top:
            return "Review"
        elif "cc" in top:
            return "CC"
        elif "authorised approver" in top or header in ["MOL", "BDM", "A1", "A2", "A3", "A4", "A5"]:
            return "Authorised Approvers"
        else:
            return "Authorised Approvers"
    
    def table_to_markdown(self, table_data: Dict) -> str:
        """
        Convert authority table to markdown sentences, separating structured data from footnotes.
        
        This is the core conversion logic that transforms table rows into readable markdown.
        The output format is designed for optimal RAG retrieval.
        
        Args:
            table_data: Dictionary containing raw_data key with table rows
            
        Returns:
            Formatted markdown string
        """
        rows = table_data["raw_data"]
        if len(rows) < 2:
            return ""
        
        header_row_1 = rows[0]
        header_row_2 = rows[1]
        lines = []
        current_main = None

        for row in rows[2:]:
            if len(row) < 3:
                continue
                
            # Check if this row contains structured data or footnotes
            if self._is_footer_row(row):
                continue  # Skip footer rows, they'll be handled separately
                
            classification = (row[1] or "").strip()
            sub_classification = (row[2] if len(row) > 2 else "").strip()
            line_index_for_approvers = None

            # Only process rows with proper serial numbers or structured classifications
            # Also process rows that contain structured table data (even without serial numbers)
            if (self._has_serial_number(row) or 
                self._is_structured_classification(classification) or 
                self._is_structured_table_row(row)):
                if classification:
                    current_main = classification.strip()
                    lines.append(f"- **{current_main}**")
                    line_index_for_approvers = len(lines) - 1

                if sub_classification:
                    level, cleaned = self._strip_bullet(sub_classification)
                    if level == 2:
                        lines.append(f"    - **{cleaned}**")
                        line_index_for_approvers = len(lines) - 1
                    elif level == 3:
                        lines.append(f"        - {cleaned}")
                        line_index_for_approvers = len(lines) - 1
                    else:
                        if cleaned and not cleaned.startswith("**"):
                            lines.append(f"    - **{cleaned}**")
                        else:
                            lines.append(f"    - {cleaned}")
                        line_index_for_approvers = len(lines) - 1

                approver_roles = {
                    "Authorised Approvers": [],
                    "Co-Management Dept.": [],
                    "Deliberation": [],
                    "Report": [],
                    "Review": [],
                    "CC": [],
                }

                for i in range(3, len(row)):
                    cell_value = (row[i] or "").strip()
                    if not cell_value:
                        continue

                    # Find effective header (handles merged or blank top cells)
                    effective_top_header = self._get_effective_top_header(header_row_1, i)

                    if self.is_numbered_marker(cell_value) or cell_value in ["●", "•"]:
                        approver_name = None
                        if i < len(header_row_2) and header_row_2[i] and header_row_2[i].strip():
                            approver_name = header_row_2[i].strip()
                        elif i < len(header_row_1) and header_row_1[i] and header_row_1[i].strip():
                            approver_name = header_row_1[i].strip()

                        if approver_name:
                            category = self.categorize_approver(approver_name, effective_top_header)
                            approver_roles[category].append(approver_name)
                        continue

                    # Handle Email logic specially
                    if cell_value.lower() == "email":
                        # If under 'Report' or spanning from 'Report' column
                        if "report" in effective_top_header.lower():
                            report_target = None
                            if i < len(header_row_2) and header_row_2[i] and header_row_2[i].strip():
                                report_target = header_row_2[i].strip()
                            else:
                                # Search around for possible A-level or MM
                                for j in range(max(0, i - 3), min(len(header_row_2), i + 4)):
                                    val = (header_row_2[j] or "").strip()
                                    if val and re.match(r"^(A[1-5]|MM|GPM)$", val):
                                        report_target = val
                                        break
                            approver_roles["Report"].append(f"{report_target or 'MM'} via Email")

                        else:
                            # Not under Report, so likely 'A4' approver row case → report to previous A-level
                            prev_target = None
                            for j in range(i - 1, 2, -1):
                                val = (row[j] or "").strip()
                                if re.match(r"^A[1-5]$", val):
                                    prev_target = val
                                    break
                            approver_roles["Report"].append(f"{prev_target or 'MM'} via Email")
                        continue

                    # All other text cells → use normal category mapping
                    category = self.categorize_approver(cell_value, effective_top_header)
                    approver_roles[category].append(cell_value)


                if any(approver_roles.values()) and line_index_for_approvers is not None:
                    approver_parts = []
                    for k, v in approver_roles.items():
                        if v:
                            unique_values = []
                            seen = set()
                            for val in v:
                                if val not in seen:
                                    unique_values.append(val)
                                    seen.add(val)
                            approver_parts.append(f"{k}: {', '.join(unique_values)}")
                    
                    if approver_parts:
                        approver_text = "; ".join(approver_parts)
                        lines[line_index_for_approvers] += f" — {approver_text}."

        return "\n".join(lines)
    
    def _is_footer_row(self, row: List[str]) -> bool:
        """Check if a row looks like a footer/note instead of structured data."""
        if not row or len(row) < 1:
            return False

        row_text = " ".join([cell for cell in row if cell and cell.strip()]).strip()
        if not row_text:
            return False

        # If first cell is a serial number → not footer
        if self._has_serial_number(row):
            return False

        # If row contains table markers (●, •, etc.) → structured, not footer
        if self._contains_table_symbols(row_text):
            return False

        # Otherwise, treat as footer if it's descriptive text
        return self._is_footer_text(row_text)

    
    def _is_structured_table_row(self, row: List[str]) -> bool:
        """Check if a row contains structured table data (not footnotes)."""
        if not row:
            return False
    
        row_text = " ".join([cell for cell in row if cell and cell.strip()]).strip()
        
        # If there are any table symbols anywhere in the row => structured
        if self._contains_table_symbols(row_text):
            return True
        
        # If sub-classification cell begins with a known bullet (e.g., '', '•', '-', etc.), treat as structured
        if len(row) > 2:
            sub = (row[2] or "").strip()
            if sub:
                _, cleaned = self._strip_bullet(sub)
                if cleaned and cleaned != sub:
                    return True
        
        # If any later column (approval columns) contains a single marker like '●' or '●(28)', treat as structured
        for cell in row[3:]:
            if not cell:
                continue
            cell_s = str(cell).strip()
            if re.search(r'[●•]', cell_s) or re.search(r'^[●•]\s*\(\d+\)', cell_s):
                return True
        
        # Check for specific structured content patterns
        structured_content_patterns = [
            r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]\s+[A-Z]',  # Circled number + capital letter
            r'^[A-Z]?[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]\s+[A-Z]',  # A① + capital letter
            r'^\(\d+\)\s+[A-Z]',  # (1) + capital letter
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+',  # Title Case + Title Case
            r'^[A-Z][a-z]+\s+[a-z]+\s+[A-Z][a-z]+',  # Multi-word title case
        ]
        
        for pattern in structured_content_patterns:
            if re.match(pattern, row_text):
                return True
        
        # Check for rows that contain approval-related content
        approval_keywords = [
            "Authorised Approvers", "Co-Management Dept", "Deliberation", "Report", "Review",
            "MOL", "BDM", "A1", "A2", "A3", "A4", "A5", "GPM", "MM", "Email"
        ]
        
        keyword_count = sum(1 for keyword in approval_keywords if keyword in row_text)
        if keyword_count >= 1:  # If it contains any approval-related terms, it's likely structured data
            return True
        
        # Check for rows that contain structured classification keywords
        classification_keywords = [
            "Matters related to", "Appointment of", "Conclusion of", "Renewal of", "Amendment of",
            "Cancellation of", "Employment", "Termination of", "Decision of", "Payment of",
            "Leasing", "Purchase", "Sale", "Transfer", "Establishment", "Closure",
            "Revision", "Abolishment", "Promotion", "Demotion", "Disciplinary",
            "Corporate tax", "Goods and service tax", "Drawdown", "Repayment",
            "Setting up", "Extension", "Borrowing", "Lending", "Foreign exchange",
            "Interest swap", "Currency swap", "Charter in", "Charter out", "Novation",
            "Annual Review", "Cargo delivery", "Master Agreement", "Annual Plan",
            "Spot Voyage", "Time Charterer", "COA", "RCA", "CVC", "Agencies Management",
            "Handling B/L", "Handling electronic", "EUA", "EU-ETS", "Forward Contracts",
            "Marine safety", "Coordination", "Honourable vessel", "Capital expenditure",
            "Supervisory agreements", "Ship manager", "BBC", "Owned vessels"
        ]
        
        for keyword in classification_keywords:
            if keyword in row_text:
                return True
        
        return False

    def _get_effective_top_header(self, header_row_1: List[str], idx: int) -> str:
        """
        Return the nearest non-empty top header for column `idx`.
        Prefer leftwards (so a spanning header on the left is captured).
        """
        if not header_row_1:
            return ""
        n = len(header_row_1)
        if 0 <= idx < n and header_row_1[idx] and header_row_1[idx].strip():
            return header_row_1[idx].strip()
        # search left
        for j in range(idx - 1, -1, -1):
            if header_row_1[j] and header_row_1[j].strip():
                return header_row_1[j].strip()
        # search right (fallback)
        for j in range(idx + 1, n):
            if header_row_1[j] and header_row_1[j].strip():
                return header_row_1[j].strip()
        return ""

    
    def _contains_table_symbols(self, text: str) -> bool:
        """Check if text contains table symbols that indicate structured data."""
        if not text:
            return False
        
        # Expanded list of table symbols (include the white-square bullet seen in PDF)
        table_symbols = ['●', '•', '◦', '▪', '▫', '◾', '']
        symbol_count = sum(1 for symbol in table_symbols if symbol in text)
        
        # If there is at least one table symbol, treat as structured (previously required >=2)
        if symbol_count >= 1:
            return True
        
        # Check for patterns like "●(15)" which are table data (numbered markers)
        if re.search(r'[●•]\s*\(\d+\)', text):
            return True
        
        # Check patterns like '● ● ●' (redundant but kept)
        if re.search(r'●\s*●\s*●', text):
            return True
        
        return False

    
    def _has_serial_number(self, row: List[str]) -> bool:
        """Check if a row has a proper serial number in the first column."""
        if not row or not row[0]:
            return False
        
        first_cell = str(row[0]).strip()
        
        # Check for various serial number patterns
        serial_patterns = [
            r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+$',  # Circled numbers
            r'^[A-Z]?[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+$',  # A①, B②, etc.
            r'^\(\d+\)$',  # (1), (2), etc.
            r'^\d+$',  # 1, 2, 3, etc.
            r'^[A-Z]\d+$',  # A1, B2, etc.
        ]
        
        for pattern in serial_patterns:
            if re.match(pattern, first_cell):
                return True
        
        return False
    
    def _is_structured_classification(self, classification: str) -> bool:
        """Check if a classification is structured data rather than descriptive text."""
        if not classification or len(classification.strip()) < 3:
            return False
        
        classification = classification.strip()
        
        # Check for structured classification patterns
        structured_patterns = [
            r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]',  # Starts with circled number
            r'^[A-Z]?[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]',  # A①, B②, etc.
            r'^\(\d+\)',  # Starts with (1), (2), etc.
            r'^[A-Z][a-z]+\s+[A-Z]',  # Title Case patterns
            r'^[A-Z][a-z]+\s+[a-z]+\s+[A-Z]',  # Multi-word title case
        ]
        
        for pattern in structured_patterns:
            if re.match(pattern, classification):
                return True
        
        # Pure structural heuristics (no keyword list):
        structural_patterns = [
            # Title Case phrase with a preposition: Action of/for/to ... Noun
            r'^[A-Z][a-z]+\s+(of|for|to|in|on|at)\s+[A-Z]',
            # Two- or three-word Title Case (e.g., Payment Approval, Vessel Purchase)
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,2}$',
            # Title with qualifier in parentheses (e.g., Lease (7), Option (III))
            r'^[A-Z][A-Za-z\s]+\((\d+|[ivxlcdmIVXLCDM]+)\)$'
        ]
        for pattern in structural_patterns:
            if re.search(pattern, classification):
                return True
        
        return False
    
    def extract_footers_and_notes(self, table_data: Dict) -> List[str]:
        """Extract footers, notes, and additional information from table data."""
        footers = []
        rows = table_data["raw_data"]

        if not rows:
            return footers

        for row in rows[2:]:  # Skip header rows
            if len(row) < 1:
                continue

            first_cell = (row[0] or "").strip()
            if not first_cell:
                continue

            # ✅ Skip pure numbers, circled numbers, A1/B2 markers, or (1)/(2)
            if self._is_pure_number_or_index(first_cell):
                continue

            # ✅ If first column has meaningful text, keep it
            if self._contains_meaningful_text(first_cell):
                cleaned_text = self._clean_footer_text(first_cell)
                if cleaned_text and len(cleaned_text.strip()) > 5:
                    footers.append(cleaned_text)

        # Remove duplicates
        seen = set()
        unique_footers = []
        for footer in footers:
            if footer not in seen:
                seen.add(footer)
                unique_footers.append(footer)

        return unique_footers
    
    def _is_pure_number_or_index(self, text: str) -> bool:
        """Check if text is purely a number or index marker."""
        if not text:
            return True
        
        text = text.strip()
        
        # Check for pure numbers
        if re.match(r'^\d+$', text):
            return True
        
        # Check for parenthetical numbers like (1), (2), etc.
        if re.match(r'^\(\d+\)$', text):
            return True
        
        # Check for circled numbers
        if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+$', text):
            return True
        
        # Check for letter-number combinations like A1, B2, etc.
        if re.match(r'^[A-Z]\d+$', text):
            return True
        
        # Check for letter-circled number combinations like A①, B②, etc.
        if re.match(r'^[A-Z][①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+$', text):
            return True
        
        return False
    
    def _contains_meaningful_text(self, text: str) -> bool:
        """Check if text contains meaningful descriptive content (not just numbering)."""
        if not text or len(text.strip()) < 3:
            return False
        
        text = text.strip()
        
        # If it's purely a number or index, it's not meaningful text
        if self._is_pure_number_or_index(text):
            return False
        
        # Check for meaningful text patterns
        meaningful_patterns = [
            r'^[A-Z][a-z]',  # Starts with capital letter followed by lowercase
            r'^[A-Z][a-z]+\s+[A-Z]',  # Title Case patterns
            r'^[A-Z][a-z]+\s+[a-z]+\s+[A-Z]',  # Multi-word title case
            r'^[A-Z][a-z]+\s+[a-z]+',  # Title case with lowercase
        ]
        
        for pattern in meaningful_patterns:
            if re.match(pattern, text):
                return True
        
        # Structural heuristics (no keyword list): treat as meaningful if it resembles a sentence
        sentence = text.strip()
        tokens = sentence.split()
        if len(tokens) >= 6 and sentence[0].isupper():
            verbs = {"is","are","was","were","has","have","had","will","shall","should","could","can","may","might",
                     "require","requires","required","include","includes","including","exclude","excludes","excluding",
                     "approve","approved","refer","referred","calculate","calculated"}
            articles_preps = {"a","an","the","to","for","of","in","on","by","from","with","under","subject","based"}
            lower = [t.lower().strip(",.;:") for t in tokens]
            if any(t in verbs for t in lower) or any(t in articles_preps for t in lower):
                return True
        
        # If it contains multiple words and is not just numbering, it's likely meaningful
        words = text.split()
        if len(words) >= 2:
            # Check if it's not just numbers and symbols
            non_numeric_words = [word for word in words if not re.match(r'^[\d\(\)①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+$', word)]
            if len(non_numeric_words) >= 1:
                return True
        
        return False
    
    def _clean_footer_text(self, text: str) -> str:
        """Clean and format footer text."""
        if not text:
            return ""
        
        # Remove extra spaces and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common formatting issues
        text = re.sub(r'-\s*-', '-', text)  # Fix double dashes
        text = re.sub(r'\s*-\s*', ' - ', text)  # Normalize dash spacing
        
        # Remove table symbols that shouldn't be in footnotes
        text = re.sub(r'[●•◦▪▫]+', '', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    
    def _is_footer_text(self, text: str) -> bool:
        """Check if text appears to be a footer or note (without hardcoded keywords)."""
        if not text or len(text.strip()) < 10:
            return False

        text = text.strip()

        # 1. Exclude pure markers or numbers (structured data)
        if re.match(r'^\d+$', text):
            return False
        if re.match(r'^\(\d+\)$', text):
            return False
        if re.match(r'^[A-Z]\d+$', text):
            return False
        if re.match(r'^[①-⑳]+$', text):  # circled numbers
            return False
        if re.match(r'^[●•◦▪▫]+$', text):
            return False

        # 2. Likely footer if sentence-like (multiple words, has verbs/punctuation)
        words = text.split()
        if len(words) >= 5:  # at least 5 words → meaningful
            return True

        # 3. Likely footer if it contains punctuation typical of notes
        if any(ch in text for ch in [":", ";", ".", "%"]):
            return True

        return False

    
    def _extract_page_footers(self, page_num: int, file_path: str) -> List[str]:
        """Extract footers from the page text around the table."""
        footers = []
        
        if not file_path:
            return footers
        
        try:
            with pdfplumber.open(file_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    text = page.extract_text()
                    
                    if text:
                        lines = text.split('\n')
                        
                        # Look for footer patterns in the last few lines of the page
                        for line in lines[-15:]:  # Check last 15 lines
                            line = line.strip()
                            if self._is_footer_text(line):
                                footers.append(line)
                        
                        # Also look for lines that contain department responsibilities
                        for line in lines:
                            line = line.strip()
                            if any(dept in line.lower() for dept in ['dept.', 'department', 'responsible for']):
                                if self._is_footer_text(line):
                                    footers.append(line)
                        
                        # Look for specific insurance-related footers
                        for line in lines:
                            line = line.strip()
                            if any(insurance_term in line.lower() for insurance_term in [
                                'business planning dept', 'ship management dept', 'hr dept',
                                'vessel-related insurances', 'tcl', 'dth', 'fdd',
                                'hull & machinery', 'loss of hire', 'war risk', 'cargo claim indemnity', 'p&i insurance'
                            ]):
                                if self._is_footer_text(line):
                                    footers.append(line)
                        
                        # Look for numbered footnotes
                        for line in lines:
                            line = line.strip()
                            if re.search(r'^\d+\s+', line) or re.search(r'\(\d+\)', line):
                                if self._is_footer_text(line):
                                    footers.append(line)
        
        except Exception as e:
            logger.debug(f"Error extracting page footers for page {page_num}: {e}")
        
        return footers
    
    def extract_page_additional_content(self, page_num: int, file_path: str) -> str:
        """Extract additional content (footers, notes, text outside tables) from a page."""
        additional_content = []
        
        if not file_path:
            return ""
        
        try:
            with pdfplumber.open(file_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    text = page.extract_text()
                    
                    if text:
                        lines = text.split('\n')
                        
                        # Extract all text content
                        all_text = []
                        for line in lines:
                            line = line.strip()
                            if line and len(line) > 3:  # Skip very short lines
                                all_text.append(line)
                        
                        # Get table boundaries to identify content outside tables
                        tables = page.extract_tables()
                        table_content = set()
                        
                        # Collect all text that appears in tables
                        for table in tables:
                            for row in table:
                                for cell in row:
                                    if cell:
                                        cell_text = str(cell).strip()
                                        if cell_text:
                                            table_content.add(cell_text)
                        
                        # Find content that's NOT in tables
                        non_table_content = []
                        for line in all_text:
                            # Check if this line is likely table content
                            is_table_content = False
                            for table_text in table_content:
                                if line in table_text or table_text in line:
                                    is_table_content = True
                                    break
                            
                            # Also check for table-specific patterns
                            if (re.search(r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]', line) or
                                re.search(r'^[●•◦▪▫]+', line) or
                                re.search(r'^[A-Z]\d+$', line) or
                                re.search(r'^\d+$', line) or
                                re.search(r'^\([0-9]+\)$', line)):
                                is_table_content = True
                            
                            if not is_table_content and len(line) > 10:  # Only meaningful content
                                non_table_content.append(line)
                        
                        # Filter and clean the additional content
                        for content in non_table_content:
                            if self._is_additional_content_meaningful(content):
                                cleaned_content = self._clean_additional_content(content)
                                if cleaned_content:
                                    additional_content.append(cleaned_content)
        
        except Exception as e:
            logger.debug(f"Error extracting additional content for page {page_num}: {e}")
        
        return "\n".join(additional_content)
    
    def _is_additional_content_meaningful(self, content: str) -> bool:
        """Heuristic check for meaningful non-table content without hardcoded keywords.

        Filters out page markers and OCR-spaced garbage; keeps short title-like lines.
        """
        if not content or len(content.strip()) < 3:
            return False

        text = content.strip()

        # Exclude page markers like "Page 5" or "Page 5 of 28"
        if re.search(r'\bpage\s+\d+(?:\s+of\s+\d+)?\b', text, re.IGNORECASE):
            return False

        # Exclude pure numbers, bare (n), or only bullets/symbols
        if re.match(r'^\d+$', text):
            return False
        if re.match(r'^\([0-9]+\)$', text):
            return False
        if re.match(r'^[●•◦▪▫\-\*\s]+$', text):
            return False

        # Exclude OCR-spaced lines: high ratio of single-character tokens
        tokens = text.split()
        if tokens:
            single_char_tokens = sum(1 for t in tokens if len(t) == 1)
            if single_char_tokens / max(len(tokens), 1) > 0.5:
                return False

        # Prefer short, title-like lines (e.g., "Head of Department")
        if 1 <= len(tokens) <= 6:
            long_tokens = sum(1 for t in tokens if len(t) >= 3)
            init_caps = sum(1 for t in tokens if t[:1].isupper())
            if long_tokens >= max(1, int(0.6 * len(tokens))) and init_caps >= max(1, int(0.5 * len(tokens))):
                return True

        # Otherwise require at least a few non-numeric tokens to keep
        non_numeric = [w for w in tokens if not re.match(r'^[\d\(\)●•◦▪▫]+$', w)]
        return len(non_numeric) >= 3

    def _clean_additional_content(self, content: str) -> str:
        """Clean and format additional content."""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Fix common formatting issues
        content = re.sub(r'-\s*-', '-', content)  # Fix double dashes
        content = re.sub(r'\s*-\s*', ' - ', content)  # Normalize dash spacing
        
        # Remove excessive table symbols that shouldn't be in additional content
        content = re.sub(r'[●•◦▪▫]+', '', content)
        
        # Clean up multiple spaces
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content