"""
Committee Table Processor

Handles extraction and processing of committee structure tables from PDFs.
Based on the original committee_processor.py for the Cosmos Embedding Admin Panel.

This module processes committee organization tables that contain:
- Committee names
- Role assignments (Chairperson, Secretariat, Members, Sub-members)
- Department/position mappings
- Symbols: ◎ (Chairperson), ● (Secretariat), ○ (Member), △ (Sub-member)
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber
from loguru import logger

logger = logging.getLogger(__name__)


class CommitteeTableProcessor:
    """Processor for committee table format PDFs (matrix with symbols ◎, ●, ○, △)."""

    SYMBOL_TO_ROLE = {
        "◎": "Chairperson",
        "●": "Secretariat",
        "○": "Member",
        "△": "Sub-member",
    }

    def __init__(self):
        pass

    def _clean_cell(self, cell: Any) -> str:
        """Clean cell content from PDF table extraction."""
        text = str(cell).strip().replace("\n", " ") if cell else ""
        return self.fix_spaced_text(text)

    def fix_spaced_text(self, text: str) -> str:
        """
        Enhanced method to clean role/department names.
        Handles OCR-spaced text and normalizes formatting.
        """
        if not text:
            return ""
        text = text.strip()
        text = text.replace("–", "-")  # normalize dash
        if " " not in text:
            return text
        if "-" in text or "/" in text:
            parts = re.split(r'([-/])', text)
            cleaned_parts = []
            for part in parts:
                if part in ['-', '/']:
                    cleaned_parts.append(part)
                else:
                    cleaned_parts.append(part.replace(' ', ''))
            return ''.join(cleaned_parts)
        else:
            return text.replace(' ', '')

    def detect_committee_format(self, file_path: str) -> bool:
        """
        Detect if PDF likely contains committee tables.
        
        Uses a combination of:
        - Keyword matching (committee, chairman, member, etc.)
        - Symbol detection (◎, ●, ○, △)
        - Department name patterns
        - Exclusion of revision/history tables
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            True if committee format detected, False otherwise
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:10], start=1):
                    try:
                        text = (page.extract_text() or "").lower()
                    except Exception:
                        text = ""

                    committee_keywords = [
                        "committee", "chairman", "chairperson", "vice chairman",
                        "member", "secretary", "secretariat", "meeting"
                    ]
                    has_committee_keyword = any(keyword in text for keyword in committee_keywords)

                    tables = page.extract_tables()
                    has_committee_structure = False
                    for table in tables:
                        if not table or len(table) < 3:
                            continue
                        table_text = " ".join([str(cell) for row in table for cell in row if cell]).lower()
                        role_indicators = ["chairman", "chairperson", "member",
                                           "secretary", "secretariat", "◎", "●", "○", "△"]
                        role_count = sum(1 for indicator in role_indicators if indicator in table_text)
                        has_symbols = any(sym in table_text for sym in ["◎", "●", "○", "△"])
                        has_department_names = any(dept in table_text for dept in [
                            "chiefexecutiveofficer", "corporatedepartments", "commercialdepartments",
                            "fleetandenvironmentalstrategy", "operations", "marine", "shipmanagement",
                            "humancapital", "informationtechnology", "generaladministration"
                        ])
                        is_revision_table = any(term in table_text for term in [
                            "enactment", "revision", "october", "may", "september", "april", "november", "july"
                        ])

                        if (has_committee_keyword and has_symbols and has_department_names and not is_revision_table):
                            has_committee_structure = True
                            logger.info(f"Committee table detected on page {page_num}")
                            break

                    if has_committee_keyword and has_committee_structure:
                        return True
        except Exception as e:
            logger.error(f"Error detecting committee format: {e}")
        return False

    def extract_tables(self, file_path: str, save_dir: Optional[Path] = None) -> List[Dict]:
        """
        Extract committee tables from PDF.
        
        Args:
            file_path: Path to the PDF file
            save_dir: Optional directory (not used for embeddings)
            
        Returns:
            List of dictionaries containing table data with metadata
        """
        all_docs = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables, start=1):
                    clean_table = [[self._clean_cell(c) for c in row]
                                   for row in table if any(self._clean_cell(c) for c in row)]
                    if not clean_table or len(clean_table) < 3:
                        continue
                    doc = {
                        "id": f"committee_p{page_num}_t{t_idx}",
                        "page": page_num,
                        "table_index": t_idx,
                        "raw_data": clean_table
                    }
                    all_docs.append(doc)
        return all_docs

    def table_to_markdown(self, table_data: Dict) -> str:
        """
        Convert committee matrix-like table to a readable markdown format.
        Handles both two-header format (Page 4) and single-header format (Page 5).
        
        The output format is optimized for RAG retrieval with clear role assignments.
        
        Args:
            table_data: Dictionary containing raw_data key with table rows
            
        Returns:
            Formatted markdown string
        """
        rows = table_data.get("raw_data", [])
        if not rows or len(rows) < 3:
            return ""

        md_lines = []

        # Skip document revision tables
        table_text = " ".join([str(cell) for row in rows for cell in row if cell]).lower()
        if any(term in table_text for term in ["enactment", "revision", "october", "may", "september", "april", "november", "july"]):
            return ""

        # Build department mapping from the two header rows
        department_mapping = {}  # col_index -> (main_heading, department)
        
        # Find the main headings row (first row with meaningful content)
        main_headings_row = None
        for i, row in enumerate(rows[:3]):
            if not row:
                continue
            row_text = " ".join([str(cell) for cell in row if cell]).upper()
            if any(heading in row_text for heading in ["EXECUTIVEOFFICERS", "GLOBAL/REGIONALDIRECTORS", "GLOBAL/REGIONAL DIRECTORS"]):
                main_headings_row = i
                break
        
        # Find department names row (row with actual department names)
        department_row = None
        for i, row in enumerate(rows):
            if not row:
                continue
            row_text = " ".join([str(cell) for cell in row if cell]).lower()
            
            # Check for department names in this row
            dept_count = sum(1 for dept in [
                "chiefexecutiveofficer", "corporatedepartments", "commercialdepartments",
                "fleetandenvironmentalstrategy", "operations", "marine", "shipmanagement",
                "humancapital", "informationtechnology", "generaladministration",
                "businessgrowth", "enterprisetransformation", "internalaudit", "groupplanning",
                "globalhumanresources", "groupaccounting", "dx&applicationsolution",
                "infrastructureandcybersecurity", "shipmanagementaccounting", "fleetstrategy",
                "environmentalstrategy", "businessstrategy", "globalstrategicallocation",
                "chartering", "operationsdept", "postfixture", "technical"
            ] if dept in row_text)
            
            # For two-header format, we need fewer department names since they're spread out
            min_dept_count = 2 if main_headings_row is not None else 3
            
            if dept_count >= min_dept_count:  # Found enough department names
                department_row = i
                break

        if department_row is None:
            logger.warning("No department row found or insufficient departments.")
            return ""

        # Build department mapping
        if main_headings_row is not None:
            # Two-header format: build mapping from main headings and departments
            main_headings = [str(cell).strip() if cell else "" for cell in rows[main_headings_row]]
            departments = [str(cell).strip() if cell else "" for cell in rows[department_row]]
            
            # Build mapping for each column
            current_main_heading = ""
            for col_idx in range(max(len(main_headings), len(departments))):
                # Get main heading for this column (may span multiple columns)
                if col_idx < len(main_headings) and main_headings[col_idx]:
                    current_main_heading = main_headings[col_idx]
                
                # Get department for this column
                department = departments[col_idx] if col_idx < len(departments) else ""
                
                if department:  # Only include columns with department names
                    department_mapping[col_idx] = (current_main_heading, department)
        else:
            # Single-header format: just departments
            departments = [str(cell).strip() if cell else "" for cell in rows[department_row]]
            for col_idx, department in enumerate(departments):
                if department:
                    department_mapping[col_idx] = ("", department)

        # Process committee rows (rows after department row)
        for row_idx in range(department_row + 1, len(rows)):
            row = rows[row_idx]
            if not row or len(row) < 2:
                continue

            # Find committee name (usually in column 1 or 2)
            committee_name = ""
            for col_idx in range(min(3, len(row))):
                cell = str(row[col_idx]).strip() if row[col_idx] else ""
                if cell and "committee" in cell.lower():
                    committee_name = cell
                    break
            
            if not committee_name:
                continue

            # Map symbols to departments with proper formatting
            chairperson, secretariat, members, sub_members = [], [], [], []
            
            # For two-header format, symbols might not be in exact department columns
            # Look for symbols in the row and map them to the nearest department
            for col_idx in range(len(row)):
                cell = str(row[col_idx]).strip() if row[col_idx] else ""
                
                if not cell or not any(sym in cell for sym in ["◎", "●", "○", "△"]):
                    continue
                
                # Find the corresponding department for this column
                if col_idx in department_mapping:
                    main_heading, department = department_mapping[col_idx]
                else:
                    # For two-header format, symbols might be in slightly different columns
                    # Find the nearest department column
                    nearest_dept_col = None
                    min_distance = float('inf')
                    
                    for dept_col in department_mapping.keys():
                        distance = abs(col_idx - dept_col)
                        if distance < min_distance and distance <= 2:  # Allow up to 2 columns difference
                            min_distance = distance
                            nearest_dept_col = dept_col
                    
                    if nearest_dept_col is not None:
                        main_heading, department = department_mapping[nearest_dept_col]
                    else:
                        continue  # Skip if no nearby department found
                
                # Format department name with heading if available
                if main_heading:
                    formatted_dept = f"{department} ({main_heading})"
                else:
                    formatted_dept = department
                
                # Check for role symbols
                if "◎" in cell:
                    chairperson.append(formatted_dept)
                elif "●" in cell:
                    secretariat.append(formatted_dept)
                elif "○" in cell:
                    members.append(formatted_dept)
                elif "△" in cell:
                    sub_members.append(formatted_dept)

            # Generate markdown in the expected format
            committee_lines = []
            
            if chairperson:
                for dept in chairperson:
                    committee_lines.append(f"◎ {dept} is the Chairperson of the {committee_name}.")
            
            if members:
                committee_lines.append(f"○ Members of the {committee_name} are: {', '.join(members)}.")
            
            if sub_members:
                committee_lines.append(f"△ Sub-members of the {committee_name} are: {', '.join(sub_members)}.")
            
            if secretariat:
                for dept in secretariat:
                    committee_lines.append(f"● Head of Department is the Secretariat of the {dept}.")

            if committee_lines:
                md_lines.append(f"**{committee_name}**")
                md_lines.extend(committee_lines)
                md_lines.append("")  # Add blank line between committees

        return "\n".join(md_lines)
    
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
        """Check if additional content is meaningful (not just page numbers/OCR noise)."""
        if not content or len(content.strip()) < 5:
            return False
        
        content_lower = content.lower()
        
        # Skip common non-meaningful content
        skip_patterns = [
            r'^page \d+$',
            r'^\d+$',
            r'^[a-z] \d+$',  # Single letter + number
            r'^\([0-9]+\)$',  # Just numbers in parentheses
            r'^[●•◦▪▫]+$',  # Just symbols
            r'^s/n$',
            r'^classification$',
            r'^authorised approver$',
            r'^co-mgmt$',
            r'^deliberation$',
            r'^report$',
            r'^review$',
            r'^cc$'
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return False
        
        # Reject page markers like "Page 5" or "Page 5 of 28"
        if re.search(r'page\s+\d+(?:\s+of\s+\d+)?', content_lower):
            return False
        # Reject pure numbers/symbols
        if re.match(r'^\d+$', content):
            return False
        if re.match(r'^[●•◦▪▫\-\*\s]+$', content):
            return False
        # Reject OCR-spaced lines dominated by single characters
        tokens = content.split()
        if tokens:
            single_char_tokens = sum(1 for t in tokens if len(t) == 1)
            if single_char_tokens / max(len(tokens), 1) > 0.5:
                return False
        # Accept bullet/numbered lines with words
        if re.match(r'^(?:[-•\*]|\d+\.)\s+\w+', content):
            return True
        # Accept short title-like lines
        if 1 <= len(tokens) <= 6:
            long_tokens = sum(1 for t in tokens if len(t) >= 3)
            init_caps = sum(1 for t in tokens if t[:1].isupper())
            if long_tokens >= max(1, int(0.6 * len(tokens))) and init_caps >= max(1, int(0.5 * len(tokens))):
                return True
        # Accept sentence-shaped lines
        if re.match(r'^[A-Z].+[a-z].+$', content) and len(tokens) >= 4:
            return True
        
        # Check for descriptive text patterns
        if (len(content.split()) >= 3 and  # At least 3 words
            not re.match(r'^[A-Z]+\d*$', content) and  # Not just letters+numbers
            not re.match(r'^[●•◦▪▫\s]+$', content)):  # Not just symbols
            return True
        
        return False
    
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
