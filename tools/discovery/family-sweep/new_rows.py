"""Rows for the families an opcode-space sweep found outside every known row.

THE IDEA
  The manifest rows come from the envytools table. An opcode-space sweep
  (gen_opcode.py, family-sweep.sh --space opcode) asks the disassembler
  about every top-16-bit prefix, and some words it accepts match no row:
  instruction families, or parts of families, the table never lists. Each
  such family becomes a row of its own, so inference can solve its layout
  like any other row. A row is a mask and value over the top 16 bits; the
  bits below are left to inference, which finds any opcode bits there by
  flipping them.

THE ALGORITHM
  1 Per prefix, collect the accepted words no row matches, by base mnemonic
    (the name before the first dot). Words the disassembler rejects, or
    prints as INVALID<n>, are answers too: they say the space is invalid.
  2 Per mnemonic M, the target set T is the prefixes with such words of M.
    A prefix may also sit inside M's row when every accepted word in it
    already matches a known row (the known row comes first in table order
    and keeps those words). Any other prefix is foreign: unmatched words of
    another mnemonic, or no accepted word at all.
  3 Start from the cube of the bits all of T agree on. If the cube holds a
    foreign prefix, cut it on its highest free bit and solve each half on
    its part of T, until each cube holds no foreign prefix. A single prefix
    shared by two mnemonics cannot be cut on the top 16 bits; it becomes
    one row naming both, and inference separates them below bit 48.
  4 Rows are numbered from FIRST (4000 by default), past every envytools
    line number, in mnemonic then value order.

usage: new_rows.py ROWS_TSV OUT_TSV RUN_DIR [RUN_DIR..]
  RUN_DIR  opcode-space run dirs (words.tsv and nv.tsv)
  OUT_TSV  decoder-rows-discovered.tsv: the manifest's columns, table
           'discovered', the baseline at a PT guard, and the evidence
           (prefixes, words) instead of a source line.
  FIRST=N in the environment changes the first row number.
"""
import collections, os, re, sys

ROWS, OUT, RUNS = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for l in open(ROWS):
    if l.startswith('#'): continue
    f = l.rstrip('\n').split('\t')
    if f[1] == 'root' and f[2] != 'OOPS': rows.append((int(f[4], 16), int(f[3], 16)))


def rejected(t): return t.startswith('!') or 'INVALID' in t
def mnemonic(t): return re.sub(r'^@!?P\w+\s+', '', t).split()[0].split('.')[0]


seen = set()
unmatched = collections.defaultdict(lambda: collections.defaultdict(int))  # prefix -> mnemonic -> words
accepted = collections.Counter()
for run in RUNS:
    for l in open(os.path.join(run, 'nv.tsv')):
        w, _, t = l.rstrip('\n').partition('\t')
        w = int(w, 16)
        if w in seen: continue
        seen.add(w)
        if rejected(t): continue
        p = w >> 48
        accepted[p] += 1
        if not any(w & m == v for m, v in rows): unmatched[p][mnemonic(t)] += 1

by_mnemonic = collections.defaultdict(set)
for p, ms in unmatched.items():
    for m in ms: by_mnemonic[m].add(p)


def foreign(p, m):
    if p in unmatched: return set(unmatched[p]) != {m}
    return accepted[p] == 0


def cubes(target, m, fixed_mask, fixed_value):
    """Cubes over target inside the cube (fixed_mask, fixed_value)."""
    agree = 0xffff
    first = next(iter(target))
    for p in target: agree &= ~(p ^ first)
    mask = (agree | fixed_mask) & 0xffff
    value = first & mask
    free = [b for b in range(15, -1, -1) if not mask >> b & 1]
    inside = [p for p in range(1 << 16) if p & mask == value] if free else [value]
    if not any(foreign(p, m) for p in inside) or not free:
        return [(mask, value, sorted(target))]
    b = free[0]
    out = []
    for bit in (0, 1):
        part = {p for p in target if (p >> b & 1) == bit}
        if part: out += cubes(part, m, fixed_mask | 1 << b, (fixed_value & ~(1 << b)) | bit << b)
    return out


result = []
for m in sorted(by_mnemonic):
    for mask, value, ps in cubes(by_mnemonic[m], m, 0, 0):
        names = sorted({x for p in ps for x in unmatched[p]})
        words = sum(unmatched[p][x] for p in ps for x in unmatched[p])
        result.append(('+'.join(names), mask, value, len(ps), words))
merged = {}
for names, mask, value, n, words in result:  # a shared prefix found from both mnemonics: one row
    merged.setdefault((mask, value), (names, n, words))
row = int(os.environ.get('FIRST', 4000))
with open(OUT, 'w') as o:
    o.write('# Rows the opcode-space sweep found outside every manifest row (tools/discovery/\n'
            '# family-sweep/new_rows.py); the bits below 48 are left to inference.\n'
            '# line\ttable\tname\tvalue\tmask\tbaseline\tevidence\n')
    for (mask, value), (names, n, words) in sorted(merged.items(), key=lambda kv: (kv[1][0], kv[0][1])):
        v, mk = value << 48, mask << 48
        o.write(f'{row}\tdiscovered\t{names.lower()}\t{v:016x}\t{mk:016x}\t{v | 7 << 16:016x}\t'
                f'opcode-space sweep: {n} prefixes, {words} words\n')
        row += 1
per = collections.Counter(names for names, _, _ in merged.values())
print(f'new_rows: {len(merged)} rows over {sum(n for _, n, _ in merged.values())} prefixes; '
      + ', '.join(f'{k} {v}' for k, v in sorted(per.items())), file=sys.stderr)
