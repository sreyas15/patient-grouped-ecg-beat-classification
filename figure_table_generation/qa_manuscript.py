"""Programmatic QA of the R3 manuscript DOCX: cross-references, numbering order, residual-term scan, placeholder scan."""
import sys, re, json, docx
from docx.oxml.ns import qn
path = sys.argv[1]
d = docx.Document(path); body = d.element.body
paras = []   # (kind, style, text)
for ch in body.iterchildren():
    tg = ch.tag.split('}')[1]
    if tg == 'p':
        st = ch.find(qn('w:pPr')); sty = st.find(qn('w:pStyle')).get(qn('w:val')) if st is not None and st.find(qn('w:pStyle')) is not None else ''
        paras.append(('p', sty, ''.join(t.text or '' for t in ch.iter(qn('w:t')))))
    elif tg == 'tbl':
        paras.append(('t', 'tbl', ' | '.join(''.join(t.text or '' for t in tc.iter(qn('w:t'))) for tc in ch.iter(qn('w:tc')))))
full = '\n'.join(t for _, _, t in paras)
rep = {}
# captions
tcaps = [int(m.group(1)) for k, s, t in paras for m in [re.match(r'Table (\d+):', t)] if m and s == 'MDPI41tablecaption']
fcaps = [int(m.group(1)) for k, s, t in paras for m in [re.match(r'Figure (\d+):', t)] if m and s == 'MDPI51figurecaption']
rep['table_captions'] = tcaps; rep['figure_captions'] = fcaps
rep['tables_sequential'] = tcaps == list(range(1, len(tcaps) + 1)); rep['figures_sequential'] = fcaps == list(range(1, len(fcaps) + 1))
rep['n_tables_in_body'] = sum(1 for k, _, _ in paras if k == 't'); rep['n_drawings'] = len(body.findall('.//' + qn('w:drawing')))
# eq numbers
eqs = []
for ch in body.iterchildren():
    if ch.tag == qn('w:p') and (ch.findall('.//' + qn('m:oMath')) or ch.findall('.//' + qn('m:oMathPara'))):
        txt = ''.join(t.text or '' for t in ch.iter(qn('w:t'))); m = re.search(r'\((\d+)\)', txt)
        if m: eqs.append(int(m.group(1)))
rep['equation_numbers'] = eqs; rep['equations_sequential'] = eqs == list(range(1, len(eqs) + 1))
# headings
heads = [t for k, s, t in paras if s in ('MDPI21heading1', 'MDPI22heading2', 'MDPI23heading3')]
secnums = set(re.match(r'(\d+(?:\.\d+)*)\.', h).group(1) for h in heads if re.match(r'\d', h))
rep['n_headings'] = len(heads)
# reference-text excluding caption/body self refs
bodytxt = '\n'.join(t for k, s, t in paras if s != 'MDPI81references')
def refs(pat): return [(m.group(0), m.start()) for m in re.finditer(pat, bodytxt)]
bad = {}
tb = set(range(1, len(tcaps) + 1)); fg = set(range(1, len(fcaps) + 1))
refd_tables = set(); refd_figs = set()
for m in re.finditer(r'Tables? (\d+)(?: and (\d+))?', bodytxt):
    for g in m.groups():
        if g:
            refd_tables.add(int(g))
            if int(g) not in tb: bad.setdefault('table_ref_missing', []).append(m.group(0))
for m in re.finditer(r'Figures? (\d+)(?:[–-](\d+)| and (\d+))?', bodytxt):
    for g in m.groups():
        if g:
            refd_figs.add(int(g))
            if int(g) not in fg: bad.setdefault('figure_ref_missing', []).append(m.group(0))
for m in re.finditer(r'Sections? (\d+(?:\.\d+)*)', bodytxt):
    if m.group(1) not in secnums: bad.setdefault('section_ref_missing', []).append(m.group(0))
eqset = set(eqs)
for m in re.finditer(r'Eqs?\. \(?(\d+)\)?(?:[–,] ?\(?(\d+)\)?)*', bodytxt):
    for n in re.findall(r'\d+', m.group(0)):
        if int(n) not in eqset: bad.setdefault('eq_ref_missing', []).append(m.group(0))
rep['unreferenced_tables'] = sorted(tb - refd_tables); rep['unreferenced_figures'] = sorted(fg - refd_figs)
# first-mention order
def first_order(pat, cap_pat):
    seen = []
    for m in re.finditer(pat, bodytxt):
        for g in m.groups():
            if g and int(g) not in seen: seen.append(int(g))
    return seen
rep['table_first_mention_order'] = first_order(r'Tables? (\d+)(?: and (\d+))?', None)
rep['figure_first_mention_order'] = first_order(r'Figures? (\d+)', None)
rep['bad_refs'] = bad
# residual terms
terms = ['triage', 'real-time', 'real time', 'EHR', 'air-gapped', 'CPU', 'PTB', 'earlier version', 'previous version', 'Previously', 'earlier analyses', 'withdrawn', 'In the earlier',
         'tier', 'ADASYN', 'DTW', 't-SNE', 'UMAP', 'Figure 6', 'TODO', 'to be inserted', 'clinical method', 'research-grade', 'clinically interpretable', 'quantitative framework',
         'screening', 'deploy', 'Keras', 'Grad-CAM', 'cardiologist', 'clinical pipeline', 'accessible', 'scalable', 'SDG', 'inline', '9.006', 'Measure-Command', 'KMeans', 'silhouette',
         'dead-filter', 'Table 18', 'Table 19', 'Gemini', 'Claude', 'compatible', 'runtime', 'latency', 'browser', 'upload', 'demonstration', 'PTB-XL']
hits = {}
for t in terms:
    ctx = []
    for m in re.finditer(re.escape(t), bodytxt, flags=re.I if t.islower() else 0):
        ctx.append(bodytxt[max(0, m.start() - 60): m.end() + 60].replace('\n', ' '))
    if ctx: hits[t] = ctx
rep['term_hits'] = hits
# generic checks
rep['placeholder_hits'] = re.findall(r'\[[^\]]*(?:insert|TBD|TODO|to be)[^\]]*\]', bodytxt, flags=re.I) + re.findall(r'\{[A-Z]:[a-zA-Z0-9_]+\}|\[@\d', bodytxt)
rep['double_spaces'] = len(re.findall(r'[a-z]  [a-z]', bodytxt))
rep['glued_words'] = sorted(set(re.findall(r'\b[a-z]+(?:based|aware|stakes|field|derived)\b', bodytxt)))[:20]
rep['words'] = len(bodytxt.split())
json.dump(rep, open(path.replace('.docx', '_QA.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
for k, v in rep.items():
    if k == 'term_hits':
        print('--- term hits'); [print(f'  [{t}] x{len(c)}: ' + ' || '.join(c[:6])) for t, c in v.items()]
    else: print(k, ':', v)
