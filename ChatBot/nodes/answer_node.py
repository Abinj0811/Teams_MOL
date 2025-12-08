# nodes/answer_node.py
import imp
from nodes.base_node import BaseNode
import re
import json

class AnswerNode(BaseNode):
    def __init__(self):
        super().__init__("AnswerNode")
    
    @staticmethod
    def replace_abbreviations(text: str) -> str:
        """
        Replace abbreviations in the text with their full form.
        Only replaces whole-word occurrences, safe for punctuation.
        Example: 'GAF' → 'GAF (Group Accounting & Finance)'
        """
        # Load abbreviation dictionary
        with open("utils/abbreviations.json", "r", encoding="utf-8") as f:
            ABBR_MAP = json.load(f)
        # Sort keys by length (longest first) to avoid partial replacements
        sorted_abbr = sorted(ABBR_MAP.keys(), key=len, reverse=True)

        for abbr in sorted_abbr:
            full = ABBR_MAP[abbr]

            # Regex for whole-word matching
            pattern = r"\b" + re.escape(abbr) + r"\b"

            # Replacement format: ABBR (Full Form)
            repl = f"{abbr} ({full})"

            # Perform replacement
            text = re.sub(pattern, repl, text)

        return text
    def format_answer(self, text: str) -> str:
        """
        Format the answer to make headings bold and improve readability.
        - Converts markdown headings (#, ##, ###) to bold text
        - Detects and formats section headings (lines ending with colon)
        - Ensures proper spacing between sections
        - Makes the text more user-friendly
        """
        if not text:
            return text
        
        lines = text.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines (will add spacing later)
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Convert markdown headings to bold
            if stripped.startswith('#'):
                # Remove # symbols and make it bold
                heading_text = re.sub(r'^#+\s*', '', stripped)
                if heading_text:
                    formatted_lines.append('')  # Add space before heading
                    formatted_lines.append(f'**{heading_text}**')
                    continue
            
            # Detect section headings: lines ending with colon that are followed by content
            # and are relatively short (likely headings, not regular sentences)
            if (stripped.endswith(':') and
                len(stripped) < 100 and
                not stripped.startswith('**') and
                not stripped.startswith('*') and
                not stripped.startswith('-') and
                not stripped.startswith('•') and
                i < len(lines) - 1):
                # Check if next line has content (not empty, not another heading)
                next_stripped = lines[i+1].strip() if i + 1 < len(lines) else ''
                if next_stripped and not next_stripped.startswith('**') and not next_stripped.startswith('#'):
                    formatted_lines.append('')  # Add space before heading
                    formatted_lines.append(f'**{stripped}**')
                    continue
            
            # Keep the line as is
            formatted_lines.append(line)
        
        # Join lines and clean up excessive blank lines
        formatted_text = '\n'.join(formatted_lines)
        # Replace 3+ consecutive newlines with 2 newlines
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
        # Remove leading/trailing whitespace
        formatted_text = formatted_text.strip()
        
        return formatted_text
    def execute(self, state: dict) -> dict:
        """
        Final formatting and only writer to final_answer.
        Reads draft_answer and evaluation metadata.
        """
        draft = state.get("draft_answer", "")
        # Format the answer to make it user-friendly with bold headings
        expanded_text = self.replace_abbreviations(draft)
        final = self.format_answer(expanded_text)
        return {"final_answer": final}