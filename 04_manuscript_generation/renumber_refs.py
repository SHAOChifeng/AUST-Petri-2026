#!/usr/bin/env python3
"""
Post-processor: renumber references in the generated OCPPP manuscript docx
so that citation numbers follow strict first-appearance order (MDPI style).

Usage: python3 renumber_refs.py <manuscript.docx>

The script:
 1. scans all paragraphs (body + table cells) in document order,
 2. records first appearance of each cited number,
 3. rewrites every citation group [a,b] with the new numbers,
 4. rewrites the References list in the new order.
"""
import re, sys
from docx import Document
from docx.shared import Pt

CIT_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

def main(path):
    doc = Document(path)

    # Gather all paragraphs in document order: body paragraphs + table cells.
    ordered = []
    for p in doc.paragraphs:
        ordered.append(p)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    ordered.append(p)

    # Pass 1: first-appearance order.
    mapping = {}  # old -> new
    next_num = 1
    for p in ordered:
        for m in CIT_RE.finditer(p.text):
            for tok in m.group(1).split(","):
                old = int(tok.strip())
                if old not in mapping:
                    mapping[old] = next_num
                    next_num += 1

    print(f"Renumber map ({len(mapping)} refs):")
    for old in sorted(mapping):
        print(f"  {old} -> {mapping[old]}")

    def repl(m):
        nums = [mapping[int(t.strip())] for t in m.group(1).split(",")]
        return "[" + ",".join(str(n) for n in nums) + "]"

    # Pass 2: rewrite citations in every paragraph run (body + tables).
    # Replace within each run individually so that per-run formatting
    # (e.g., a bold lead-in followed by regular text) is preserved.
    def rewrite_paragraph(p):
        changed = False
        for r in p.runs:
            new = CIT_RE.sub(repl, r.text)
            if new != r.text:
                r.text = new
                changed = True
        if not p.runs and CIT_RE.search(p.text):
            p.add_run(CIT_RE.sub(repl, p.text))
            changed = True
        return changed

    n_changed = 0
    for p in ordered:
        if rewrite_paragraph(p):
            n_changed += 1
    print(f"Rewrote {n_changed} paragraphs with citations.")

    # Pass 3: rewrite the References list.
    # Locate the References heading and the Abbreviations heading.
    ref_idx = None
    end_idx = None
    for i, p in enumerate(doc.paragraphs):
        if ref_idx is None and p.text.strip().lower() == "references":
            ref_idx = i
        if ref_idx is not None and end_idx is None and i > ref_idx and \
           p.text.strip().lower().startswith("abbreviation"):
            end_idx = i
            break
    if ref_idx is None:
        print("ERROR: References heading not found")
        return 1

    refs = {}  # old -> entry text (without leading number)
    for p in doc.paragraphs[ref_idx + 1:end_idx]:
        m = re.match(r"^(\d+)\.\s+(.*)$", p.text.strip())
        if m:
            refs[int(m.group(1))] = m.group(2)

    # Reference paragraphs (in order of the old numbering).
    ref_paras = [p for p in doc.paragraphs[ref_idx + 1:end_idx]
                 if re.match(r"^\d+\.\s", p.text.strip())]

    if len(refs) != len(ref_paras):
        print(f"WARNING: parsed {len(refs)} refs but found {len(ref_paras)} paragraphs")

    # New order: entries sorted by new number.
    new_order = sorted(mapping.items(), key=lambda kv: kv[1])
    for (old, new), para in zip(new_order, ref_paras):
        entry = refs.get(old, f"[missing ref {old}]")
        for r in list(para.runs):
            r._element.getparent().remove(r._element)
        r = para.add_run(f"{new}. {entry}")
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)

    doc.save(path)
    print(f"Saved renumbered document: {path}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                   else "output/OCPPP_Agronomy_v21.docx"))
