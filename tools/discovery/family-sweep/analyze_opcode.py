"""Classify an opcode-space sweep (gen_opcode.py words, nv.tsv, sb.tsv).

Writes into RUN_DIR:
  new_prefixes.tsv   prefix  mnemonics  sb_kinds  example_word  nvdisasm_text
                     prefixes nvdisasm decodes that no root manifest row
                     covers (undiscovered families), sorted by mnemonic.
  new_families.tsv   mnemonic  prefixes  sb_kinds  prefix_list  example_word
                     new_prefixes clustered by base mnemonic (before '.').
  unknown_known.tsv  row  row_name  mnemonic  words  example_word  text
                     nvdisasm decodes, shadowbox says Unknown, and the word
                     matches a manifest row.
  row_gap.tsv        same columns, words that decode, sit in a prefix some
                     row covers, yet match no row (opcode bits below 48).
  wrong_admissions.tsv  prefix  row  sb_kind  words  example_word  reject
                     nvdisasm rejects, shadowbox decodes.
  split.tsv          prefix  prefix_row  mnemonics  per-anchor mnemonics
                     prefixes whose decoded words disagree on the base
                     mnemonic: opcode bits below 48, follow up with a split.
  report.txt         counts and the top new families.
usage: analyze_opcode.py RUN_DIR
"""
import sys, re, os, collections
R = sys.argv[1].rstrip('/') + '/'
nv = {}; sb = {}
for l in open(R + 'nv.tsv'):
    w, t = l.rstrip('\n').split('\t', 1); nv[w] = t
for l in open(R + 'sb.tsv'):
    w, k = l.rstrip('\n').split('\t'); sb[w] = k
names = {}
for l in open(os.environ['ROWS_TSV']):
    if l.startswith('#'): continue
    f = l.split('\t'); names.setdefault(f[0], f[2])
def norm(t): return re.sub(r'^@!?P[T0-6]\s+', '', t)
def mnem(t): s = norm(t).split(); return s[0] if s else '?'
def base(t): return mnem(t).split('.')[0]
pref = collections.defaultdict(list)   # prefix -> [(tag, word, eff, text, kind)]
prow = {}
for l in open(R + 'words.tsv'):
    p, eff, w, tag, pr = l.rstrip('\n').split('\t')
    prow[p] = pr; pref[p].append((tag, w, eff, nv.get(w, '!MISSING'), sb.get(w, '?')))
dec = lambda t: not t.startswith('!')
npre = nrow = 0; newp = []; uk = collections.defaultdict(list); gap = collections.defaultdict(list)
wa = collections.defaultdict(list); split = []; nwords = sum(len(v) for v in pref.values())
ndec = sum(1 for v in pref.values() for x in v if dec(x[3]))
for p in sorted(pref):
    ws = pref[p]; d = [x for x in ws if dec(x[3])]
    for tag, w, eff, t, k in ws:
        if dec(t) and k == 'Unknown':
            (uk if eff != '-' else gap)[(eff if eff != '-' else p, base(t))].append((w, t))
        if not dec(t) and k not in ('Unknown', '?'):
            wa[(p, eff, k)].append((w, t[len('!REJECT '):] if t.startswith('!REJECT') else t))
    if not d: continue
    npre += 1
    bs = sorted({base(x[3]) for x in d})
    if len(bs) > 1:
        split.append((p, prow[p], ','.join(bs), ' '.join(f'{x[0]}={base(x[3])}' for x in d)))
    if prow[p] == 'row': nrow += 1; continue
    ms = sorted({mnem(x[3]) for x in d}); ks = sorted({x[4] for x in d})
    newp.append((bs[0], p, ','.join(ms), ','.join(ks), d[0][1], norm(d[0][3])))
newp.sort()
with open(R + 'new_prefixes.tsv', 'w') as f:
    f.write('#prefix\tmnemonics\tsb_kinds\texample_word\tnvdisasm_text\n')
    for b, p, ms, ks, w, t in newp: f.write(f'0x{p}\t{ms}\t{ks}\t{w}\t{t}\n')
fam = collections.defaultdict(list)
for b, p, ms, ks, w, t in newp: fam[b].append((p, ks, w))
fams = sorted(fam.items(), key=lambda x: -len(x[1]))
with open(R + 'new_families.tsv', 'w') as f:
    f.write('#mnemonic\tprefixes\tsb_kinds\tprefix_list(first 64)\texample_word\n')
    for b, ps in fams:
        ks = sorted({k for _, kk, _ in ps for k in kk.split(',')})
        f.write(f'{b}\t{len(ps)}\t{",".join(ks)}\t{",".join("0x"+p for p, _, _ in ps[:64])}\t{ps[0][2]}\n')
def grouped(name, g, rowcol):
    with open(R + name, 'w') as f:
        f.write('#row\trow_name\tmnemonic\twords\texample_word\ttext\n')
        for (r, b), xs in sorted(g.items(), key=lambda x: -len(x[1])):
            f.write(f'{r}\t{names.get(r, "-") if rowcol else "prefix"}\t{b}\t{len(xs)}\t{xs[0][0]}\t{norm(xs[0][1])}\n')
grouped('unknown_known.tsv', uk, True); grouped('row_gap.tsv', gap, False)
with open(R + 'wrong_admissions.tsv', 'w') as f:
    f.write('#prefix\trow\tsb_kind\twords\texample_word\treject\n')
    for (p, e, k), xs in sorted(wa.items(), key=lambda x: (-len(x[1]), x[0])):
        f.write(f'0x{p}\t{e}\t{k}\t{len(xs)}\t{xs[0][0]}\t{xs[0][1]}\n')
with open(R + 'split.tsv', 'w') as f:
    f.write('#prefix\tprefix_row\tmnemonics\tper_anchor\n')
    for s in split: f.write('0x' + '\t'.join(s) + '\n')
L = [f'WORDS {nwords}  decoded by nvdisasm {ndec}  prefixes swept {len(pref)}',
     f'PREFIXES nvdisasm decodes: {npre}  covered by a manifest row: {nrow}  new: {npre - nrow}',
     f'NEW FAMILIES {len(fams)} (new_families.tsv): ' + ', '.join(f'{b}x{len(ps)}' for b, ps in fams[:15]),
     f'UNKNOWN_KNOWN words {sum(map(len, uk.values()))} in {len(uk)} row/mnemonic groups;'
     f' ROW_GAP words {sum(map(len, gap.values()))} in {len(gap)} groups',
     f'WRONG ADMISSIONS words {sum(map(len, wa.values()))} in {len(wa)} prefix/kind groups',
     f'SPLIT prefixes (low-bit opcode bits) {len(split)}: '
     + ', '.join(f'0x{s[0]}({s[2]})' for s in split[:20])]
for q in ('e003', 'e0c3'):
    d = [x for x in pref.get(q, []) if dec(x[3])]
    L.append(f'PREFIX 0x{q}: prefix_row={prow.get(q)} decoded {len(d)}/{len(pref.get(q, []))} '
             + ','.join(sorted({mnem(x[3]) for x in d})) + ' sb=' + ','.join(sorted({x[4] for x in pref.get(q, [])})))
open(R + 'report.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
