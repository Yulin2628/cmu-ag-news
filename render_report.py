import re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

root = Path(__file__).resolve().parent
font_pairs = [
    ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
]
for regular, bold in font_pairs:
    if Path(regular).exists() and Path(bold).exists():
        pdfmetrics.registerFont(TTFont('Arial', regular))
        pdfmetrics.registerFont(TTFont('Arial-Bold', bold))
        break
else:
    raise RuntimeError('Install Arial or DejaVu Sans before rendering the report.')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyTextCustom', fontName='Arial',fontSize=9.5,leading=14,spaceAfter=7))
styles.add(ParagraphStyle(name='TitleCustom', fontName='Arial-Bold',fontSize=23,leading=29,spaceAfter=15))
styles.add(ParagraphStyle(name='HeadingCustom', fontName='Arial-Bold',fontSize=14,leading=19,spaceBefore=13,spaceAfter=8,keepWithNext=True))
styles.add(ParagraphStyle(name='SubheadingCustom', fontName='Arial-Bold',fontSize=11,leading=15,spaceBefore=10,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='TableCustom', fontName='Arial',fontSize=8.2,leading=11.2))
styles.add(ParagraphStyle(name='CodeCustom', fontName='Arial',fontSize=8,leading=11,leftIndent=10,rightIndent=10,spaceAfter=5,textColor=colors.HexColor('#334155')))


def inline(text):
    text = text.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    text = escape(text)
    text = re.sub(r'`([^`]+)`',r'<font color="#334155">\1</font>',text)
    text = re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    return text


story = []
lines = (root/'report.md').read_text(encoding='utf-8').splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line.startswith('```'):
        code = []
        i += 1
        while i < len(lines) and not lines[i].startswith('```'):
            if lines[i].strip():
                code.append(Paragraph(escape(lines[i]),styles['CodeCustom']))
            i += 1
        story += code
    elif line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch('[-: ]+',c) for c in cells):
                rows.append([Paragraph(inline(c),styles['TableCustom']) for c in cells])
            i += 1
        if rows:
            width = A4[0]-96
            cols = len(rows[0])
            weights = [1]*cols
            if cols >= 6:
                weights[0] = 1.9
                if cols == 7:
                    weights[3] = 1.6
            widths = [width*w/sum(weights) for w in weights]
            tab = Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            tab.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2e8f0')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#64748b')),
                ('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#cbd5e1')),
            ]))
            story += [tab,Spacer(1,10)]
        continue
    elif line.startswith('!['):
        match = re.match(r'!\[([^]]*)\]\(([^)]+)\)',line)
        if match:
            path = root / match[2]
            from PIL import Image as PILImage
            with PILImage.open(path) as image:
                width,height=image.size
            render_width = min(A4[0]-96,450)
            image_block = Image(str(path),width=render_width,height=render_width*height/width)
            caption_index = i + 1
            while caption_index < len(lines) and not lines[caption_index].strip():
                caption_index += 1
            if caption_index < len(lines) and lines[caption_index].strip().startswith(('Figure ', 'Rows are')):
                caption = Paragraph(inline(lines[caption_index].strip()),styles['BodyTextCustom'])
                story.append(KeepTogether([image_block, Spacer(1,4), caption]))
                i = caption_index
            else:
                story.append(image_block)
    elif line.startswith('# '):
        story.append(Paragraph(inline(line[2:]),styles['TitleCustom']))
    elif line.startswith('## '):
        if line == '## References':
            references = [Paragraph('References',styles['HeadingCustom'])]
            references += [Paragraph(inline(item.strip()),styles['BodyTextCustom'])
                           for item in lines[i+1:] if item.strip()]
            story.append(KeepTogether(references))
            break
        story.append(Paragraph(inline(line[3:]),styles['HeadingCustom']))
    elif line.startswith('### '):
        story.append(Paragraph(inline(line[4:]),styles['SubheadingCustom']))
    elif line.startswith('> '):
        story.append(Paragraph(inline(line[2:]),styles['CodeCustom']))
    else:
        story.append(Paragraph(inline(line),styles['BodyTextCustom']))
    i += 1


def footer(canvas,doc):
    canvas.setFont('Arial',8)
    canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(48,27,'AG News: in-context learning and LoRA')
    canvas.drawRightString(A4[0]-48,27,str(doc.page))


doc = SimpleDocTemplate(str(root/'report.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=43,
                        title='News classification with in-context examples and LoRA',author='')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(root/'report.pdf')
