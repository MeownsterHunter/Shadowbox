"""Diff nvdisasm against shadowbox, cluster, rank, write the report.

nvdisasm text containing INVALID<n> is a reserved encoding: treated as a
rejection, so shadowbox decoding it is a wrong admission.
Buckets:
  missing  - nvdisasm prints an instruction, shadowbox says Unknown
             (undiscovered family members, or whole unimplemented rows);
  wrong    - shadowbox decodes, nvdisasm rejects (wrong admissions);
  mismatch - both decode, but the shadowbox kind is not one KIND_OPS
             allows for that nvdisasm mnemonic (or the kind is new).
Missing words are clustered by nvdisasm mnemonic+modifiers plus operand
shape (register numbers and immediates normalised; RZ/PT/.H1/|..|/-
kept). Cluster categories:
  new-operands      mnemonic decodes elsewhere; this operand shape doesn't
  new-modifiers     mnemonic+modifier set never decodes anywhere
  same-text         identical disassembly decodes from another word: only
                    don't-care bits differ (decoder stricter than hardware)
  operand-values    the nearest decoded word differs only in register,
                    predicate or attribute operand bits (e.g. R0 vs RZ)
  row-unimplemented shadowbox decodes nothing in that manifest row
Rank = tier(mnemonic) * category weight * log2(1+count). Tier 3: the
graphics forms (conversions, IPA, texture, FSET/FSETP, half ops, LDC,
ALD/AST, SHFL...); tier 2: common ALU/memory; tier 1: everything else.
usage: analyze.py ROWS_JSON ANCHORS_JSON RUNDIR
"""
import sys, re, json, collections, math, os
rows = json.load(open(sys.argv[1])); A = json.load(open(sys.argv[2])); R = sys.argv[3].rstrip('/') + '/'
TIER3 = set('F2F I2F F2I I2I IPA TEX TEXS TLD TLDS TLD4 TLD4S TMML TXQ TXD TXA FSET FSETP HADD2 HMUL2 HFMA2 '
            'HSET2 HSETP2 HADD2_32I HMUL2_32I HFMA2_32I LDC ALD AST SHFL OUT KIL PIXLD SULD SUST FSWZADD MUFU '
            'S2R CS2R AL2P ISBERD DEPBAR FCMP FMNMX'.split())
TIER2 = set('FADD FMUL FFMA FADD32I FMUL32I FFMA32I IADD IADD3 IADD32I ISCADD ISCADD32I IMAD IMUL XMAD LOP LOP3 '
            'LOP32I SHL SHR SHF BFE BFI SEL MOV MOV32I PRMT ISETP ISET ICMP IMNMX PSETP PSET CSETP R2P P2R '
            'LDS STS LDL STL LDG STG LD ST BRA SSY SYNC BRK PBK CONT PCNT EXIT POPC FLO LEA RRO DSET DSETP '
            'DADD DMUL DFMA DMNMX VOTE MEMBAR BAR ATOM ATOMS RED'.split())
CATW = {'operand-values': 1, 'new-operands': 3, 'new-modifiers': 3, 'row-unimplemented': 1, 'same-text': 1}
# shadowbox kind -> nvdisasm base mnemonics it may stand for (validated 2026-10-18 sweep)
KIND_OPS = {'Atomic': 'ATOMS', 'BitFieldExtract': 'BFE', 'BitFieldInsert': 'BFI', 'Branch': 'BRA',
 'BrxRegister': 'BRX', 'CSetP': 'CSETP', 'Call': 'CAL', 'Convert': 'F2F F2I I2F I2I', 'DoubleFma': 'DFMA',
 'DoubleMultiply': 'DMUL', 'DoubleAdd': 'DADD', 'DoubleMinMax': 'DMNMX', 'DSet': 'DSET', 'DSetP': 'DSETP',
 'Exit': 'EXIT', 'FAdd': 'FADD FADD32I', 'FCmp': 'FCMP', 'FFma': 'FFMA FFMA32I', 'FMul': 'FMUL FMUL32I',
 'FSet': 'FSET', 'FSetP': 'FSETP', 'FMnmx': 'FMNMX', 'Flo': 'FLO', 'FunnelShift': 'SHF', 'GlobalAtomic': 'ATOM RED',
 'GlobalMemory': 'LDG STG', 'GroupMemoryBarrier': 'MEMBAR', 'MemoryBarrier': 'MEMBAR',
 'HalfArithmetic': 'HADD2 HADD2_32I HFMA2 HFMA2_32I HMUL2 HMUL2_32I', 'HalfSet': 'HSET2 HSETP2',
 'IAdd': 'IADD', 'IAdd3': 'IADD3', 'ICmp': 'ICMP', 'ISet': 'ISET', 'ISetP': 'ISETP', 'IMnmx': 'IMNMX',
 'IntegerMul': 'IMAD IMUL IMAD32I IMUL32I', 'IsbeRead': 'ISBERD', 'Kill': 'KIL', 'Ldc': 'LDC',
 'LocalLoad': 'LDL', 'LocalStore': 'STL', 'SharedLoad': 'LDS', 'SharedStore': 'STS', 'Lop': 'LOP LOP32I',
 'Lop3': 'LOP3', 'Mov': 'MOV MOV32I', 'MuFu': 'MUFU', 'Nop': 'NOP DEPBAR', 'P2R': 'P2R', 'R2P': 'R2P',
 'PSet': 'PSET', 'PSetP': 'PSETP', 'PopJump': 'BRK CONT SYNC', 'PushTarget': 'PBK PCNT', 'Schedule': 'SSY',
 'Popc': 'POPC', 'Prmt': 'PRMT', 'Rro': 'RRO', 'S2R': 'S2R CS2R', 'Sel': 'SEL', 'ShiftAdd': 'ISCADD ISCADD32I IADD32I',
 'Shl': 'SHL', 'Shr': 'SHR', 'Shuffle': 'SHFL', 'Surface': 'SULD SUST', 'SurfaceAtomic': 'SUATOM SURED',
 'SwizzleAdd': 'FSWZADD', 'Texture': 'TEX TEXS TLD TLD4 TLD4S TLDS TMML TXQ TXD', 'VideoMinMax': 'VMNMX',
 'VideoMultiplyAdd': 'VMAD', 'VideoSetP': 'VSETP', 'VideoAdd': 'VADD', 'Vote': 'VOTE', 'VoteVtg': 'VOTE', 'Xmad': 'XMAD',
 'Lea': 'LEA', 'IndexedPatchLoad': 'ALD', 'PatchAttributeLoad': 'ALD', 'AttributeLoad': 'ALD',
 'OutputAttributeLoad': 'ALD', 'PhysicalAttributeLoad': 'ALD', 'PhysicalOutputAttributeLoad': 'ALD',
 'UnmappedAttributeLoad': 'ALD', 'AttributeStore': 'AST', 'AttributeAddress': 'AL2P',
 'IpaPass': 'IPA', 'IpaPassMulW': 'IPA', 'IpaSaturate': 'IPA', 'OutsideBranch': 'BRA', 'OutsideSchedule': 'SSY',
 'OutsidePushTarget': 'PBK PCNT', 'LeaHi': 'LEA', 'IntegerMulAdd': 'IMAD IMADSP IMAD32I XMAD',
 'FMnMx': 'FMNMX', 'FMnMxSelect': 'FMNMX', 'IMinMax': 'IMNMX', 'IMinMaxSelect': 'IMNMX', 'DoubleSetP': 'DSETP',
 'DoubleSet': 'DSET', 'DoubleToInt': 'F2I', 'DoubleToFloat': 'F2F', 'DoubleRound': 'F2F', 'CSet': 'CSET', 'LopWithPredicate': 'LOP LOP32I',
 'Lop3WithPredicate': 'LOP3', 'GlobalReduce': 'RED', 'SharedCompareExchange': 'ATOMS', 'ControlBarrier': 'BAR',
 'GeometryOut': 'OUT', 'GeometryOutDynamic': 'OUT', 'PhysicalAttributeStore': 'AST', 'SurfaceReduce': 'SURED',
 'Reconverge': 'SYNC', 'Brx': 'BRX', 'OutsideBrx': 'BRX', 'ResolvedSchedule': 'SSY', 'ResolvedPushTarget': 'PBK PCNT',
 'MisalignedBranch': 'BRA', 'MisalignedSchedule': 'SSY', 'MisalignedPushTarget': 'PBK PCNT', 'OutsideCall': 'CAL',
 'MisalignedCall': 'CAL', 'Padding': 'NOP'}
KIND_OPS = {k: set(v.split()) for k, v in KIND_OPS.items()}
W = {}
for l in open(R + 'words.tsv'):
    a, b, h, t = l.rstrip('\n').split('\t'); W[h] = dict(row=b, tag=t)
for l in open(R + 'nv.tsv'):
    h, t = l.rstrip('\n').split('\t', 1)
    if h in W: W[h]['nv'] = t
for l in open(R + 'sb.tsv'):
    h, t = l.rstrip('\n').split('\t'); W[h]['sb'] = t
lost = [h for h, v in W.items() if 'nv' not in v]
for h in lost: del W[h]
def norm(t): return re.sub(r'^@!?P[T0-6]\s+', '', t)
def mnem(t): s = norm(t).split(); return s[0] if s else '?'
def shape(t):
    s = norm(t).split(None, 1); o = s[1] if len(s) > 1 else ''
    o = re.sub(r'\bR(?!Z)\d+', 'R', o); o = re.sub(r'\bP[0-6]\b', 'P', o)
    o = re.sub(r'-?0x[0-9a-f]+|\b\d+(\.\d+)?(e[+-]\d+)?\b', '#', o); o = re.sub(r'c\[#\]\[#\]', 'c[#][#]', o)
    return o
# nvdisasm prints some reserved encodings instead of rejecting them, naming
# the field value INVALID<n> (TLDS ..., INVALID5; HSETP2.F.INVALIDBOP3;
# OUT.INVALIDOUTSTYPE0; IPA.*.INVALID sample mode). Those count as rejected.
INVALID = re.compile(r'INVALID')
for v in W.values():
    v['reserved'] = not v['nv'].startswith('!') and bool(INVALID.search(v['nv']))
    v['rej'] = v['nv'].startswith('!') or v['reserved']; v['unk'] = v['sb'] == 'Unknown'
miss = [h for h, v in W.items() if not v['rej'] and v['unk']]
wrong = [h for h, v in W.items() if v['rej'] and not v['unk']]
both = [h for h, v in W.items() if not v['rej'] and not v['unk']]
mism = [h for h in both if mnem(W[h]['nv']).split('.')[0] not in KIND_OPS.get(W[h]['sb'], ())]
acc_row = collections.defaultdict(list); acc_rm = collections.defaultdict(list)
for h in both:
    acc_row[W[h]['row']].append(int(h, 16)); acc_rm[(W[h]['row'], mnem(W[h]['nv']))].append(int(h, 16))
_rs = __import__('random').Random(1)
for d_ in (acc_row, acc_rm):
    for k_ in d_:
        if len(d_[k_]) > 3000: d_[k_] = _rs.sample(d_[k_], 3000)  # nearest-neighbour search pool
acc_text = set(norm(W[h]['nv']) for h in both)
acc_mn = collections.Counter(mnem(W[h]['nv']) for h in both)
acc_key = collections.Counter((mnem(W[h]['nv']), shape(W[h]['nv'])) for h in both)
bitname = {}
for k, v in rows.items():
    m = {}
    for n, fm in v['fields'].items():
        for b in range(64):
            if fm >> b & 1: m.setdefault(b, n.split('/')[-1] if not n.startswith('T(') else n.split('/')[0])
    for b in range(64):
        if v['unc'] >> b & 1: m.setdefault(b, 'uncovered')
        if v['mask'] >> b & 1: m.setdefault(b, 'opcode')
    bitname[k] = m
def opmask(r):
    m = 0
    for n, fm in rows[r]['fields'].items():
        if re.search(r'reg\d\d_bf|pred\d\d_bf|amem|not@', n.split('/')[-1]): m |= fm
    return m
def bitdesc(r, x):
    return ' '.join(f'{b}({bitname.get(r, {}).get(b, "?")})' for b in range(64) if x >> b & 1)
clusters = collections.defaultdict(list)
for h in miss:
    v = W[h]; r = v['row']; t = norm(v['nv']); m = mnem(v['nv'])
    if not acc_row[r]: cat = 'row-unimplemented'
    elif t in acc_text: cat = 'same-text'
    elif acc_mn[m]: cat = 'new-operands'
    else: cat = 'new-modifiers'
    clusters[(m, shape(v['nv']), r, cat)].append(int(h, 16))
def tier(m):
    b = m.split('.')[0]; return 3 if b in TIER3 else 2 if b in TIER2 else 1
out = []
for (m, sh, r, cat), ws in clusters.items():
    an = A.get(r, {}).get('anchor', rows[r]['base'] if r in rows else 0)
    ex = min(ws, key=lambda w: (W[f'{w:016x}']['tag'] != 'a', bin(w ^ an).count('1'), w))
    pool = acc_rm.get((r, m)) or acc_row[r]
    near = min(pool, key=lambda a: (bin(a ^ ex).count('1'), a)) if pool else None
    if cat in ('new-operands', 'new-modifiers') and near is not None and not ((ex ^ near) & ~opmask(r)):
        cat = 'operand-values'
    out.append(dict(score=round(tier(m) * CATW[cat] * math.log2(1 + len(ws)), 2), cat=cat, mnemonic=m, shape=sh,
        row=r, rowname=rows[r]['name'] if r in rows else '-', count=len(ws), accepted_same_key=acc_key[(m, sh)],
        example=f'{ex:016x}', text=norm(W[f'{ex:016x}']['nv']),
        nearest=f'{near:016x}' if near is not None else '', nearest_text=norm(W[f'{near:016x}']['nv']) if near is not None else '',
        nearest_kind=W[f'{near:016x}']['sb'] if near is not None else '',
        diff=bitdesc(r, ex ^ near) if near is not None else ''))
out.sort(key=lambda c: -c['score'])
def tsv(name, recs):
    with open(R + name, 'w') as f:
        if not recs: return
        f.write('\t'.join(recs[0]) + '\n')
        for c in recs: f.write('\t'.join(str(x) for x in c.values()) + '\n')
tsv('clusters.tsv', out)
with open(R + 'missing.tsv', 'w') as f:
    for h in sorted(miss): f.write(f"{h}\t{W[h]['row']}\t{norm(W[h]['nv'])}\n")
# wrong admissions grouped, with predictive bits inside the row
wg = collections.defaultdict(list)
for h in wrong:
    msg = W[h]['nv'] if not W[h]['reserved'] else '!RESERVED printed as ' + ' '.join(sorted(set(re.findall(r'[A-Z_]*INVALID[A-Z_0-9]*', W[h]['nv']))))
    wg[(W[h]['row'], W[h]['sb'], msg)].append(int(h, 16))
wout = []
pops = collections.defaultdict(list)
for h, v in W.items():
    if not v['unk']: pops[v['row']].append((int(h, 16), v['rej']))
for (r, k, msg), ws in sorted(wg.items(), key=lambda kv: -len(kv[1])):
    pop = pops[r]
    pred = []
    for b in range(64):
        on = [x for w, x in pop if w >> b & 1]; off = [x for w, x in pop if not w >> b & 1]
        if len(on) > 20 and len(off) > 20:
            d_ = sum(on) / len(on) - sum(off) / len(off)
            if abs(d_) > 0.4: pred.append(f'{b}({bitname[r].get(b, "?")})={"1" if d_ > 0 else "0"}:{abs(d_):.2f}')
    ex = sorted(ws, key=lambda w: bin(w ^ A[r]['anchor']).count('1'))[:4]
    wout.append(dict(row=r, rowname=rows[r]['name'], kind=k, nvdisasm=msg[8:], count=len(ws),
                     examples=' '.join(f'{w:016x}' for w in ex), reject_predicting_bits=' '.join(pred) or '-'))
tsv('wrong_admissions.tsv', wout)
mg = collections.defaultdict(list)
for h in mism: mg[(W[h]['sb'], mnem(W[h]['nv']).split('.')[0])].append(h)
mout = [dict(kind=k, mnemonic=m, count=len(hs), examples=' '.join(f"{h}:{norm(W[h]['nv'])}" for h in sorted(hs)[:3]))
        for (k, m), hs in sorted(mg.items(), key=lambda kv: -len(kv[1]))]
tsv('mismatch.tsv', mout)
# attribute-address coverage for AMEM rows
aout = []
for r, v in rows.items():
    af = [fm & ~v['mask'] for n, fm in v['fields'].items() if n.split('/')[-1].startswith('amem')]
    if not af: continue
    fm = af[0]; lo = min(b for b in range(64) if fm >> b & 1)
    ok = collections.defaultdict(set); gap = collections.defaultdict(set)
    for h, w in W.items():
        if w['row'] != r or w['rej']: continue
        mm = mnem(w['nv']); a = (int(h, 16) & fm) >> lo
        (gap if w['unk'] else ok)[mm].add(a)
    for mm in sorted(set(ok) | set(gap)):
        g = sorted(gap[mm] - ok[mm])
        if not g: continue
        rng_ = []; s = p = g[0]
        for x in g[1:] + [None]:
            if x is not None and x == p + 1: p = x; continue
            rng_.append(f'0x{s:x}' if s == p else f'0x{s:x}-0x{p:x}')
            if x is not None: s = p = x
        aout.append(dict(row=r, mnemonic=mm, decoded_attrs=len(ok[mm]), undecoded_attrs=len(g), ranges=','.join(rng_)))
tsv('attributes.tsv', aout)
cc = collections.Counter()
for c in out: cc[c['cat']] += c['count']
unimpl = collections.Counter()
for c in out:
    if c['cat'] == 'row-unimplemented': unimpl[(c['row'], c['rowname'])] += c['count']
L = []
L.append(f"words {len(W)}  nvdisasm-reserved(INVALID) {sum(v['reserved'] for v in W.values())}  nvdisasm-accepted {sum(not v['rej'] for v in W.values())}  shadowbox-decoded {sum(not v['unk'] for v in W.values())}  lost {len(lost)}")
L.append(f"BUCKETS  missing {len(miss)}  wrong-admission {len(wrong)}  name-mismatch {len(mism)}")
L.append('missing by category: ' + ', '.join(f'{k} {v}' for k, v in cc.most_common()))
L.append(''); L.append('TOP 15 MISSING-MEMBER CLUSTERS (mnemonic+modifiers; at most 3 per row; full list clusters.tsv)')
# one line per mnemonic+modifier set (operand shapes merged), at most 3 per row
agg = {}
for c in out:
    if c['cat'] not in ('new-operands', 'new-modifiers'): continue
    k = (c['row'], c['mnemonic'])
    if k in agg: agg[k]['count'] += c['count']; agg[k]['shapes'] += 1
    else: agg[k] = dict(c, shapes=1)
top = []; per_row = collections.Counter()
for c in sorted(agg.values(), key=lambda c: -tier(c['mnemonic']) * CATW[c['cat']] * math.log2(1 + c['count'])):
    if per_row[c['row']] >= 3: continue
    per_row[c['row']] += 1; top.append(c)
    if len(top) == 15: break
for i, c in enumerate(top, 1):
    L.append(f"{i:2}. {c['mnemonic']}  [{c['cat']}, {c['count']} words in {c['shapes']} operand shapes, row gm107.c:{c['row']} {c['rowname']}]")
    L.append(f"    example {c['example']}: {c['text']}")
    L.append(f"    nearest decoded {c['nearest']} ({c['nearest_kind']}): {c['nearest_text']}")
    L.append(f"    differing bits: {c['diff']}")
L.append(''); L.append('TOP 8 OPERAND-VALUE CLUSTERS (same modifiers decode with other register/predicate/attribute values)')
for c in [c for c in out if c['cat'] == 'operand-values'][:8]:
    L.append(f"  {c['mnemonic']} {c['shape']} row {c['row']}: {c['count']} words, e.g. {c['example']} ({c['text']}) vs {c['nearest']} ({c['nearest_text']})")
L.append(''); L.append('TOP 10 SAME-TEXT CLUSTERS (decoder rejects bits nvdisasm ignores)')
for c in [c for c in out if c['cat'] == 'same-text'][:10]:
    L.append(f"  {c['mnemonic']} row {c['row']}: {c['count']} words, e.g. {c['example']} vs {c['nearest']} differ in {c['diff']}")
L.append(''); L.append('ROWS WITH NOTHING DECODED BUT NVDISASM ACCEPTS (top 15 by words)')
L.append('  ' + ', '.join(f'{r} {n} ({c})' for (r, n), c in unimpl.most_common(15)))
L.append(''); L.append('WRONG ADMISSIONS (shadowbox decodes, nvdisasm rejects)')
for w in wout:
    L.append(f"  row {w['row']} {w['rowname']} -> {w['kind']}: {w['count']} words; nvdisasm: {w['nvdisasm']}")
    L.append(f"    examples {w['examples']}; reject-predicting bits {w['reject_predicting_bits']}")
if not wout: L.append('  none')
L.append(''); L.append('NAME MISMATCHES (both decode, kind not allowed for mnemonic)')
for m in mout: L.append(f"  {m['kind']} vs {m['mnemonic']}: {m['count']} e.g. {m['examples']}")
if not mout: L.append('  none')
L.append(''); L.append('ATTRIBUTE ADDRESSES NVDISASM ACCEPTS THAT NEVER DECODE (attributes.tsv)')
for a in sorted(aout, key=lambda a: -a['undecoded_attrs'])[:12]:
    L.append(f"  row {a['row']} {a['mnemonic']}: {a['undecoded_attrs']} undecoded vs {a['decoded_attrs']} decoded: {a['ranges'][:160]}")
L.append(''); L.append('DATA: ' + ' '.join(R + f for f in ('words.tsv', 'nv.tsv', 'sb.tsv', 'clusters.tsv', 'missing.tsv',
        'wrong_admissions.tsv', 'mismatch.tsv', 'attributes.tsv', 'report.txt')))
open(R + 'report.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
