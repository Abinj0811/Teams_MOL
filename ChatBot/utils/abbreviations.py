from bs4 import BeautifulSoup
import json
import re

def extract_abbreviations(html):
    soup = BeautifulSoup(html, "html.parser")
    result = {}

    blocks = soup.select("div.default > p")

    for block in blocks:

        # Replace <br> with actual newlines BEFORE extracting text
        for br in block.find_all("br"):
            br.replace_with("\n")

        text = block.get_text("\n")

        # Split by real line breaks
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for line in lines:
            # Must contain a hyphen separating ABBR - Full form
            if "-" not in line:
                continue

            # Split only by the first hyphen
            abbr, full = line.split("-", 1)
            abbr = abbr.strip()
            full = full.strip()

            # filter out long non-abbreviation keys
            if len(abbr) > 20:
                continue

            # Save entry
            result[abbr] = full

    return result


# Example running
def scrape_and_extract():
    html = open("abbrev_ip.html", encoding="utf-8").read()
    data = extract_abbreviations(html)

    with open("abbreviations.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Extracted", len(data), "abbreviations.")
    




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
   

text = "For the acquisition of a new product with a value of less than US$50,000, the approval process depends on whether the asset is IT-related or not. **(A) Fixed Assets (excluding IT-related assets):** - Less than US$50,000: — Authorised Approvers: A4 — Review to GPM — CC Dept: GAF **(B) IT-related Fixed Assets:** - US$25,000 or more (but less than US$50,000): — Authorised Approvers: A3 — Review to GPM — Co-Management Dept: ICS / DXS — CC Dept: GAF - US$10,000 or more (but less than US$25,000): — Authorised Approvers: A3 — Co-Management Dept: ICS / DXS — CC Dept: GAF - Less than US$10,000: — Authorised Approvers: A4 — Co-Management Dept: ICS / DXS — CC Dept: GAF **Summary:** For new product acquisitions of less than US$50,000, the most specific threshold is “Less than US$50,000” for non-IT assets and, for IT-related assets, the applicable sub-thresholds are “US$25,000 or more,” “US$10,000 or more,” and “Less than US$10,000,” each with their respective approval requirements."
updated_ans = replace_abbreviations(text)
print(updated_ans)