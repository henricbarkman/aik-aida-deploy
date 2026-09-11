"""Markdown to Word, for the report and the open sheet.

One converter, so the report and the sheet leave Aida looking the same. It reads
the subset of markdown Aida writes: headings, bullet and numbered lists, bold and
italic, tables, quotes and rules. Anything else becomes a plain paragraph, so a
line the converter does not understand still reaches the reader.
"""

from __future__ import annotations

import io
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BRAND_BLUE = RGBColor(0x4A, 0x90, 0xD9)
GRAY_66 = RGBColor(0x66, 0x66, 0x66)
GRAY_99 = RGBColor(0x99, 0x99, 0x99)
HEADER_BG = "4A90D9"


def _add_rich_runs(paragraph, text):
    """Parse inline markdown (bold, italic) into runs on a paragraph."""
    for part in re.split(r'(\*\*[^*]+?\*\*|\*[^*]+?\*)', text):
        if part.startswith('**') and part.endswith('**'):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)


def _style_header_cell(cell):
    """Apply white-on-blue header styling to a table cell."""
    tc_pr = cell._element.find(qn('w:tcPr'))
    if tc_pr is None:
        tc_pr = cell._element.makeelement(qn('w:tcPr'), {})
        cell._element.insert(0, tc_pr)
    tc_pr.append(tc_pr.makeelement(qn('w:shd'), {
        qn('w:val'): 'clear', qn('w:color'): 'auto', qn('w:fill'): HEADER_BG,
    }))
    for paragraph in cell.paragraphs:
        for r in paragraph.runs:
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r.font.size = Pt(10)


def markdown_to_docx(markdown: str) -> bytes:
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)
    for level in range(1, 4):
        doc.styles[f'Heading {level}'].font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

    header_p = doc.add_paragraph()
    header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header_p.add_run('Aida')
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = BRAND_BLUE
    run = header_p.add_run('  |  Klimatberäkning av ombyggnad')
    run.font.size = Pt(10)
    run.font.color.rgb = GRAY_66

    def rich(text, style_name=None):
        p = doc.add_paragraph(style=style_name)
        _add_rich_runs(p, text)
        return p

    active_table = None
    table_is_first_row = False
    for line in markdown.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#### '):
            active_table = None
            p = doc.add_paragraph()
            run = p.add_run(stripped[5:])
            run.bold = True
            run.font.size = Pt(11)
        elif stripped.startswith('### '):
            active_table = None
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith('## '):
            active_table = None
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith('# '):
            active_table = None
            doc.add_heading(stripped[2:], level=1)
        elif stripped.startswith('> ') or stripped == '>':
            # A quote is the source's own words: set off by an indent and italics,
            # so it cannot be read as Aida's.
            active_table = None
            p = rich(stripped[2:])
            p.paragraph_format.left_indent = Cm(1)
            for r in p.runs:
                r.italic = True
        elif stripped.startswith('- ') or stripped.startswith('* '):
            active_table = None
            rich(stripped[2:], style_name='List Bullet')
        elif re.match(r'^\d+\.\s', stripped):
            active_table = None
            rich(re.sub(r'^\d+\.\s', '', stripped), style_name='List Number')
        elif stripped.startswith('|') and '|' in stripped[1:]:
            # Split on unescaped pipes only. A "|" inside a cell is written as
            # "\|", because a row that splits into the wrong number of cells used
            # to be dropped without a word - and the rows most likely to contain
            # a pipe are the appendix rows that disclose a figure is not Aida's.
            cells = [c.strip().replace('\\|', '|') for c in re.split(r'(?<!\\)\|', stripped)[1:-1]]
            if cells and all(set(c) <= {'-', ':', ' '} for c in cells):
                continue
            if cells:
                if active_table is None:
                    active_table = doc.add_table(rows=0, cols=len(cells))
                    active_table.style = 'Table Grid'
                    active_table.alignment = WD_TABLE_ALIGNMENT.CENTER
                    table_is_first_row = True
                # Fit rather than drop. A row we cannot place is still a row the
                # reader was meant to see, so a mismatch loses at most the layout,
                # never the disclosure.
                ncols = len(active_table.columns)
                if len(cells) > ncols:
                    cells = cells[:ncols - 1] + [' '.join(cells[ncols - 1:])]
                elif len(cells) < ncols:
                    cells = cells + [''] * (ncols - len(cells))
                row = active_table.add_row()
                for i, cell_text in enumerate(cells):
                    row.cells[i].text = re.sub(r'\*\*(.+?)\*\*', r'\1', cell_text)
                    for paragraph in row.cells[i].paragraphs:
                        for r in paragraph.runs:
                            r.font.size = Pt(10)
                    if table_is_first_row:
                        _style_header_cell(row.cells[i])
                table_is_first_row = False
        elif stripped == '' or stripped.startswith('---') or stripped.startswith('***'):
            active_table = None
        else:
            active_table = None
            rich(stripped)

    doc.add_paragraph()
    footer_p = doc.add_paragraph()
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer_p.add_run('Genererad av Aida | AI-stödd klimatanalys för ombyggnadsprojekt')
    run.font.size = Pt(8)
    run.font.color.rgb = GRAY_99

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
