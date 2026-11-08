"""Render report.md with explicit page breaks using ReportLab and supplied fonts."""

import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--font-dir', type=Path, required=True)
    parser.add_argument('--cjk-font', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont('Body', str(args.font_dir / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('BodyBold', str(args.font_dir / 'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFont(TTFont('CJK', str(args.cjk_font)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
    style = ParagraphStyle('body', fontName='Body', fontSize=9.6, leading=13.3,
                           textColor=colors.HexColor('#222b36'), spaceAfter=7)
    headings = {
        1: ParagraphStyle('title', parent=style, fontName='BodyBold', fontSize=18, leading=23, spaceAfter=14),
        2: ParagraphStyle('section', parent=style, fontName='BodyBold', fontSize=13, leading=17, spaceBefore=7, spaceAfter=9),
        3: ParagraphStyle('subsection', parent=style, fontName='BodyBold', fontSize=10.5, leading=14, spaceBefore=5, spaceAfter=7),
    }
    table_style = ParagraphStyle('cell', parent=style, fontSize=8.4, leading=11.2, spaceAfter=0)
    quote_style = ParagraphStyle('quote', parent=style, leftIndent=12, borderPadding=7,
                                 backColor=colors.HexColor('#f0f4f7'), spaceBefore=3, spaceAfter=10)

    def inline(text):
        text = escape(text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" color="#245c88">\1</a>', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'([\u3400-\u9fff]+)', r'<font name="CJK">\1</font>', text)
        return text

    width = A4[0] - 92
    story = []
    lines = (args.root / 'report.md').read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
            i += 1
            continue
        heading = re.match(r'^(#{1,3}) (.+)$', line)
        if heading:
            story.append(Paragraph(inline(heading[2]), headings[len(heading[1])]))
            i += 1
            continue
        picture = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if picture:
            img = Image(str(args.root / picture[2]))
            ratio = width / img.imageWidth
            img.drawWidth, img.drawHeight = width, img.imageHeight * ratio
            story.extend([img, Spacer(1, 7)])
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append([Paragraph(inline(c), table_style) for c in cells])
                i += 1
            ncols = len(rows[0])
            first_width = width * (0.48 if ncols > 2 else 0.52)
            widths = [first_width] + [(width - first_width) / (ncols - 1)] * (ncols - 1)
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e4ebf1')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#b6c5d2')),
            ]))
            story.extend([table, Spacer(1, 9)])
            continue
        if line.startswith('> '):
            story.append(Paragraph(inline(line[2:]), quote_style))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '>', '![', '<!--')):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(' '.join(paragraph)), style))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont('Body', 8)
        canvas.setFillColor(colors.HexColor('#596777'))
        canvas.drawString(46, 28, 'Дом и квартира / Qwen2.5-3B / SFT')
        canvas.drawRightString(A4[0] - 46, 28, str(document.page))
        canvas.restoreState()

    destination = args.output or args.root / 'report.pdf'
    doc = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=46, rightMargin=46,
                            topMargin=35, bottomMargin=43, title='Дом и квартира. Эксперименты SFT',
                            author='Учебный проект home-safety-sft')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination)


if __name__ == '__main__':
    main()
