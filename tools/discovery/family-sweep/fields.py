"""Read the row manifest and field samples; classify each row's bits.

For every decoder row (one envytools gm107 OP8B line) the field samples
are one-field-at-a-time probes against a baseline word. OR-ing
(sample XOR baseline) per field name gives that field's bit mask.
Fields are then split into:
  mod  - selector tables T(...) and flags name@bit and small (<=5 bit)
         immediates outside the operand area: the bits varied jointly;
  data - registers, predicates, constant-buffer/immediate operands,
         memory offsets, branch targets, system registers, guard pred;
  unc  - bits outside the opcode mask that no sampled field touches.
         envytools treats them as don't-care; the oracle may not.
usage: fields.py DATA_DIR OUT_JSON
"""
import sys, re, collections, json
D = sys.argv[1].rstrip('/') + '/'
rows = {}
for l in open(D + 'decoder-rows.tsv'):
    if l.startswith('#'): continue
    f = l.rstrip('\n').split('\t')
    rows.setdefault(f[0], dict(line=f[0], table=f[1], name=f[2], value=int(f[3], 16),
                               mask=int(f[4], 16), base=int(f[5], 16)))
fm = collections.defaultdict(lambda: collections.defaultdict(int))
for l in open(D + 'decoder-samples.tsv'):
    if l.startswith('#'): continue
    f = l.rstrip('\n').split('\t')
    if f[1] == 'baseline': continue
    r = rows[f[0]]
    fm[f[0]][f[1].split('=')[0]] |= int(f[3], 16) ^ r['base']
DATA = re.compile(r'(REG_|_RZ_|C3\d|MEM|BTARG|sched|SYS_|PRED\d\d|^T\(pred\)$|^T\(pred39\)$|T\(pred\)/)')
def isdata(name):
    if DATA.search(name): return True
    m = re.match(r'([USFD])(\d\d)_(\d\d)', name.split('/')[-2] if '/' in name else name)
    if m:
        return int(m.group(2)) > 5 or int(m.group(3)) in (20, 28)
    return False
out = {}
for k, r in rows.items():
    if r['table'] != 'root' or r['name'] == 'OOPS': continue
    mod = data = 0
    for n, m in fm[k].items():
        if isdata(n): data |= m
        else: mod |= m
    unc = ((1 << 64) - 1) & ~r['mask'] & ~mod & ~data
    out[k] = dict(r, mod=mod & ~r['mask'], data=data, unc=unc, fields=dict(fm[k]))
json.dump(out, open(sys.argv[2], 'w'))
print(f'fields: {len(out)} rows', file=sys.stderr)
