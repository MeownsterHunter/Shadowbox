"""Generate combination words per decoder row.

Opcode bits stay fixed. Operands stay at the row anchor (and, when the
anchor moved, also at the manifest baseline). The row's modifier bits
are varied jointly: exhaustive when there are <= DEPTH of them, else
SAMPLES seeded-random words. On top of that:
  - each exhaustive combination (rows with <= 14 modifier bits) gets
    one twin with each uncovered bit set at p=0.15;
  - every pair of uncovered bits, and every uncovered bit x every single
    modifier bit, is set on the anchor;
  - attribute-address rows (AMEM fields: AST, ALD, IPA) get every
    attribute value x up to 64 modifier combinations, at both anchors
    (the baseline keeps index register R0: that is how ALD a[R0+imm]
    showed up as a wrong admission);
  - system-register rows (SYS_20: S2R, CS2R) get every register value.
Each word is attributed to the first manifest row that matches it
(table order), which is the row a decoder in table order would use.
usage: gen.py ROWS_JSON ANCHORS_JSON ROWS(all|a,b,..) DEPTH SAMPLES OUT_TSV
output: intended_row  effective_row  word_hex  anchor(a|b)
"""
import json, random, sys, itertools
d = json.load(open(sys.argv[1])); A = json.load(open(sys.argv[2]))
sel = sys.argv[3]; DEPTH = int(sys.argv[4]); N = int(sys.argv[5]); OUT = sys.argv[6]
rng = random.Random(5252)
def bits(m): return [i for i in range(64) if m >> i & 1]
order = sorted(d, key=int)
masks = [(d[k]['mask'], d[k]['value'], k) for k in order]
def first_row(w):
    for m, v, k in masks:
        if w & m == v: return k
    return '-'
rows = order if sel == 'all' else [r for r in sel.split(',') if r in d]
seen = set(); out = open(OUT, 'w'); st = open(OUT + '.stats', 'w'); total = 0
def place(base, bl, x):
    w = base
    for j, b in enumerate(bl):
        if x >> j & 1: w |= 1 << b
    return w
for k in rows:
    v = d[k]; named = v['mod']; unc = v['unc'] & ~named
    nb = bits(named); ub = bits(unc)
    words = []
    anchors = [('a', A[k]['anchor'])]
    if A[k]['anchor'] != v['base']: anchors.append(('b', v['base']))
    for tag, an in anchors:
        base = an & ~(named | unc)
        depth = DEPTH if tag == 'a' else min(DEPTH, 12)
        if len(nb) <= depth:
            for x in range(1 << len(nb)):
                w = place(base, nb, x); words.append((tag, w))
                if ub and len(nb) <= 14:
                    words.append((tag, w | sum(1 << b for b in ub if rng.random() < 0.15)))
            mode = 'exhaustive'
        else:
            n = N if tag == 'a' else 4096
            for _ in range(n):
                w = base | sum(1 << b for b in nb if rng.random() < 0.5)
                words.append((tag, w | sum(1 << b for b in ub if rng.random() < 0.1)))
            mode = 'random'
        if tag == 'a':
            words += [(tag, base | 1 << p | 1 << q) for p, q in itertools.combinations(ub, 2)]
            words += [(tag, base | 1 << p | 1 << q) for p in ub for q in nb]
        if True:  # both anchors: baseline keeps a real index register (ALD a[R0+imm])
            for fname, fm in v['fields'].items():
                last = fname.split('/')[-1]
                if not (last.startswith('amem') or last == 'sys20_r'): continue
                fb = bits(fm & ~v['mask'])
                combos = range(1 << len(nb)) if len(nb) <= 6 else \
                    [rng.getrandbits(len(nb)) for _ in range(64)] + [0]
                for x in combos:
                    w0 = place(an & ~(named | fm), nb, x)
                    for y in range(1 << len(fb)):
                        words.append((tag, place(w0, fb, y)))
    n = 0
    for tag, w in words:
        if w in seen: continue
        seen.add(w); n += 1
        out.write(f'{k}\t{first_row(w)}\t{w:016x}\t{tag}\n')
    total += n
    st.write(f'{k}\t{v["name"]}\t{len(nb)}\t{len(ub)}\t{mode}\t{n}\n')
print(f'generated {total} words over {len(rows)} rows', file=sys.stderr)
