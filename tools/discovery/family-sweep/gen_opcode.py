"""Opcode-space words: every value of bits 63..48, at neutral operands.

The row sweep (gen.py) keeps opcode bits fixed to manifest rows, so it can
only find members of families we already know. This walks the opcode space
itself: all 65,536 values of the top 16 bits (the granularity of row keys
like 0x5ca8), each with the low 48 bits at these anchors:
  z   all zeros (guard @P0, R0 everywhere);
  r   RZ (0xff) in the usual register fields Rd 0..7, Ra 8..15, Rb 20..27,
      Rc 39..46, PT (7) in the guard 16..18; the RZ bytes also cover the
      predicate fields at 0..2, 3..5 and 39..41 with PT;
  s1, s2  seeded-random low 48 bits (guard kept PT so the word is live).
Each word is tagged with the first root manifest row (table order) whose
mask/value it matches, '-' if none, and with the prefix's row status:
'row' when some root row agrees with the prefix on the row's top-16 mask
bits, '-' otherwise.
With PREFIXES_FILE (one hex prefix per line, first column), only those
prefixes are swept, and the random fills come from SEED and the prefix, so a
resweep draws new fills: more samples where an earlier sweep covered nothing.
Rows also come from decoder-rows-discovered.tsv beside DECODER_ROWS_TSV when
it exists (rows found by such sweeps), after the manifest's rows.
usage: gen_opcode.py DECODER_ROWS_TSV RANDOM_FILLS OUT_TSV [PREFIXES_FILE SEED]
output: prefix_hex  effective_row  word_hex  anchor  prefix_row(row|-)
"""
import os, random, sys
M, NR, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
PREFIXES = None
if len(sys.argv) > 5:
    PREFIXES = sorted({int(l.split()[0], 16) for l in open(sys.argv[4]) if l.strip() and not l.startswith('#')})
    SEED = int(sys.argv[5])
rows = []
for path in (M, os.path.join(os.path.dirname(M), 'decoder-rows-discovered.tsv')):
    if not os.path.exists(path): continue
    for l in open(path):
        if l.startswith('#'): continue
        f = l.rstrip('\n').split('\t')
        if f[1] not in ('root', 'discovered') or f[2] == 'OOPS': continue
        rows.append((int(f[4], 16), int(f[3], 16), f[0]))
def first_row(w):
    for m, v, k in rows:
        if w & m == v: return k
    return '-'
hm = [(m >> 48, v >> 48) for m, v, _ in rows]
RZ = 0xff | 0xff << 8 | 7 << 16 | 0xff << 20 | 0xff << 39
rng = random.Random(5253); L48 = (1 << 48) - 1
n = 0; known = 0
with open(OUT, 'w') as o:
    for p in (range(1 << 16) if PREFIXES is None else PREFIXES):
        pr = 'row' if any((p ^ v) & m == 0 for m, v in hm) else '-'
        known += pr == 'row'
        lows = [('z', 0), ('r', RZ)]
        if PREFIXES is not None: rng = random.Random(SEED << 16 | p)
        lows += [(f's{i+1}', (rng.getrandbits(48) & ~(0xf << 16)) | 7 << 16) for i in range(NR)]
        for tag, lo in lows:
            w = p << 48 | (lo & L48)
            o.write(f'{p:04x}\t{first_row(w)}\t{w:016x}\t{tag}\t{pr}\n'); n += 1
print(f'generated {n} words over {65536 if PREFIXES is None else len(PREFIXES)} prefixes ({known} prefixes compatible with a root row)', file=sys.stderr)
