"""Verify that every file frozen in locked_protocol.json is unchanged. Imported by 06_train.py and 07_score.py; aborts on any mismatch."""
import os, sys, json, hashlib
R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def verify():
    lp = os.path.join(R2, 'locked_protocol.json')
    if not os.path.exists(lp): sys.exit('ABORT: protocol not locked (locked_protocol.json missing)')
    if _sha(lp) != open(os.path.join(R2, 'locked_protocol.sha256')).read().split()[0]: sys.exit('ABORT: locked_protocol.json itself was modified')
    lock = json.load(open(lp)); bad = []
    for rel, h in lock['frozen_files'].items():
        p = rel if os.path.isabs(rel) else os.path.join(R2, rel)
        if not os.path.exists(p) or _sha(p) != h: bad.append(rel)
    if bad: sys.exit(f'ABORT: locked files changed or missing: {bad}')
    lock['lock_file_sha256'] = _sha(lp); return lock
if __name__ == '__main__':
    l = verify(); print('lock OK:', l['lock_file_sha256'], '|', len(l['frozen_files']), 'files verified')
