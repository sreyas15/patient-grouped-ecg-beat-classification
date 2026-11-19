"""Verify the archive: (1) every file in FILE_MANIFEST.csv exists with the recorded size and SHA-256; (2) locked_protocol.sha256 matches locked_protocol.json;
(3) every frozen file recorded in manifests/manifest_post_training.json (protocol_lock.frozen_files) that is present in the archive has the recorded hash;
(4) all 45 checkpoints match the hashes recorded in the locked manifest.  usage: python verify_archive.py"""
import csv, hashlib, json, os, sys
here = os.path.dirname(os.path.abspath(__file__)); bad = []; n = 0
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
for r in csv.DictReader(open(os.path.join(here, 'FILE_MANIFEST.csv'), encoding='utf-8')):
    if r['relative_path'] == 'FILE_MANIFEST.csv': continue
    p = os.path.join(here, *r['relative_path'].split('/')); n += 1
    if not os.path.exists(p): bad.append(('missing', r['relative_path'])); continue
    if os.path.getsize(p) != int(r['size_bytes']) or sha(p) != r['sha256']: bad.append(('hash', r['relative_path']))
lp = open(os.path.join(here, 'protocol', 'locked_protocol.sha256')).read().split()[0]
if lp.lower() != sha(os.path.join(here, 'protocol', 'locked_protocol.json')).lower(): bad.append(('protocol-hash', 'locked_protocol.json'))
man = json.load(open(os.path.join(here, 'manifests', 'manifest_post_training.json'), encoding='utf-8'))
pm = {r['original_relative_path_under_project_R2_tree']: r['archive_path'] for r in csv.DictReader(open(os.path.join(here, 'PATH_MAP.csv')))}
k = 0
for rel, h in man['protocol_lock']['frozen_files'].items():
    if rel in pm:
        k += 1
        if sha(os.path.join(here, *pm[rel].split('/'))) != h: bad.append(('frozen-hash', rel))
c = 0
for run in man['completed_runs']['runs']:
    for fn, v in run['files'].items():
        if fn.endswith('.pt'):
            c += 1
            if sha(os.path.join(here, 'checkpoints', run['run'], fn)) != v['sha256']: bad.append(('checkpoint', run['run'] + '/' + fn))
print(f'manifest files checked: {n}; frozen files matched to the locked manifest: {k}; checkpoints: {c}; problems: {len(bad)}')
for b in bad: print('PROBLEM', b)
sys.exit(1 if bad else 0)
