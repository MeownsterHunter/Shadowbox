"""Bit-flip inference: per-row Maxwell bit layouts from nvdisasm diffs.

Steps (family-infer.sh drives them; each reads/writes the run dir R):
  pick R DATA OPRUN    (INFER_ROWS=LINE,.. limits the rows) bases.tsv: one known-valid word per (row, text shape)
                       from the manifest samples (samples.nv.tsv) and the
                       opcode-space run's nvdisasm-valid words; flips.words.tsv:
                       every base with each of its 64 bits flipped (unique).
  pc1 R, pc2 R         pc1.words.tsv: every base, asked again at a known
                       address; pc2: pc.rel.tsv (rows and atoms whose number
                       moved with the address) and pc2.words.tsv (those rows'
                       flips and pairs, asked again with addresses).
  pairs R              pairs.words.tsv: base ^ (3 << i) for adjacent bits that
                       both act as modifier/enum/flag on that base (field
                       joining test).
  layout R             layouts.tsv, conflicts.tsv, model.json.
  compose R            validate.words.tsv: new words from the layouts.
  validate R           validate.tsv (per row) and validate-words.tsv.
  export R OUT ORACLE... OUT/decoder-formats.tsv (the table a decoder generator
                       reads; family-export.sh) and OUT/decoder-format-samples.tsv
                       (every held word inside a variant's fixed bits, with each
                       text the disassembler gave it).
All nvdisasm text lives in R/*.nv.tsv (word_hex<TAB>text or !REJECT ...).

Text model (parse): guard (absent = @PT), base mnemonic, dotted modifiers,
operands. Each operand is a shape plus atoms: R (RZ=255), P (PT=7) and
numbers; shape keeps everything else (-, |..|, !, ~, words like 2D).
Memory brackets are canonical [Rn+off] (RZ / 0x0 when printed without
them; 'c[bank]' and pure-number cbuf offsets stay as printed); the raw
zero form seen per row and operand is kept so predictions render exactly.
A text containing INVALID<n> counts as a rejection (analyze.py convention).

Flip classes (one flip of one bit against its base text), by precedence:
  opcode  base mnemonic changes;  reject  nvdisasm refuses the word;
  form    operand count, atom count or atom kind (register, predicate,
          number) changes;  mod  dotted modifiers
  change (names = tokens that appear/disappear);  enum  an operand's
  non-atom words change (1D/2D, SR_ names);  flag  only -, |, !, ~
  change on an operand;  reg  one R/P number changes by 2^k (field start
  = bit - k);  imm  one number changes (value bit k, 's' sign, 'f' float);
  none  identical text.
Row merge: a bit's role is its most common informative class (not none,
not reject); a bit only ever rejecting is opcode(invalid). Fields are
contiguous runs of one role and key (reg: operand slot+start, imm: slot);
adjacent modifier/enum/flag bits join when their pair flip is not the
sum of the single flips. Conflicts: a bit with two informative roles
(multi-role), with a role on some bases and no effect on others
(conditional), reg/imm keys that disagree (slot-shift), and a modifier
value naming two token sets (value-ambiguous).
Print rules learned from the answers, not listed: a register field printed
inverted (xor a constant); a register atom that is another field plus a
constant or the zero register, as selectors decide (derive: a CAS pair); an
immediate printed 0x0 under some selectors (zero_by), and no-effect bits
that are such an immediate under other selectors; an operand left out when
its value is one constant (the row's omit: PT, RZ, 0x0); refusals keyed by
whether an immediate is zero, backing off to single selectors for key
values never seen; modifier order a row never showed taken from the rows
printing the same mnemonic.
"""
import sys, re, json, random, collections, os, itertools, struct, math
from decimal import Decimal, Context
C = collections.Counter
R = sys.argv[2].rstrip('/') + '/' if len(sys.argv) > 2 else ''
IMPORTANT = ('mod', 'enum', 'flag')
PR, PP, PN = '\x01', '\x02', '\x03'  # atom placeholders inside shapes
def disp(s): return s.replace(PR, 'R').replace(PP, 'P').replace(PN, '#')
ORDER = ('opcode', 'reject', 'form', 'mod', 'enum', 'flag', 'reg', 'imm')

ADDR = {}  # word -> address nvdisasm printed it at (texts asked with NVADDR=1)

def load_nv(*names):
    nv = {}
    for n in names:
        if os.path.exists(R + n):
            for l in open(R + n):
                w, _, t = l.rstrip('\n').partition('\t')
                if not w: continue
                t, _, a = t.partition('\t@'); w = int(w, 16); nv[w] = t
                if a: ADDR[w] = int(a, 16)
                else: ADDR.pop(w, None)
    return nv

def load_rel():
    """row -> operand atoms ('j.n') whose number is the word's address plus a value"""
    rel = collections.defaultdict(set)
    if os.path.exists(R + 'pc.rel.tsv'):
        for l in open(R + 'pc.rel.tsv'):
            row, a = l.rstrip('\n').split('\t'); rel[row].add(a)
    return rel

def at_pc(p, x, slots):
    """p with its PC-relative numbers made relative to word x's address (None when unknown)"""
    if p is None or not slots: return p
    for a in slots:
        j, n = map(int, a.split('.'))
        if j < len(p['atoms']) and n < len(p['atoms'][j]) and p['atoms'][j][n][0] == 'N':
            if x not in ADDR: return None
            p['atoms'][j][n] = ('N', p['atoms'][j][n][1] - ADDR[x], 'PC')
    return p

ATOM = re.compile(r'(?<![A-Za-z0-9_.?])(R\d+|RZ|P\d+|PT|-?(?:0x[0-9a-f]+|\d+\.\d*(?:e[+-]?\d+)?|\d+e[+-]?\d+|[+-]?INF|[+-]?[QS]NAN|[+-]?NAN|\d+))(?![A-Za-z0-9_])')
BR = re.compile(r'(R\d+|RZ)?(?:\+?(-?0x[0-9a-f]+))?')

def split_ops(s):
    out, d, cur = [], 0, ''
    for ch in s:
        if ch in '[{': d += 1
        elif ch in ']}': d -= 1
        if ch == ',' and d == 0: out.append(cur.strip()); cur = ''
        else: cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

def canon_brackets(o, zero):
    """memory/indexed brackets -> [Rn+off]; returns new text, records raw zero form"""
    res, i = '', 0
    while True:
        j = o.find('[', i)
        if j < 0: return res + o[i:]
        k = o.find(']', j)
        if k < 0: return res + o[i:]
        inner = o[j + 1:k]; pre = o[j - 1] if j else ''
        m = BR.fullmatch(inner)
        pre2 = o[max(0, j - 2):j]  # c[bank] [offset]: some forms print the space
        keep = pre == 'c' or not m or not inner or ((pre == ']' or pre2 == '] ') and not m.group(1))
        if keep: res += o[i:k + 1]
        else:
            r, off = m.group(1) or 'RZ', m.group(2) or '0x0'
            if r == 'RZ' and off == '0x0': zero.add(inner)
            res += o[i:j] + f'[{r}+{off}]'
        i = k + 1

def atomval(a):
    if a == 'RZ': return 'R', 255
    if a == 'PT': return 'P', 7
    if a[0] == 'R': return 'R', int(a[1:])
    if a[0] == 'P': return 'P', int(a[1:])
    s = a.lstrip('-')
    neg = a.startswith('-')
    if s.startswith('0x'): v = int(s, 16)
    elif s.isdigit(): v = int(s)
    else: return 'F', a
    return 'N', -v if neg else v

def parse(text, zero=None):
    if text.startswith('!') or 'INVALID' in text: return None
    zero = set() if zero is None else zero
    t = text
    g = '@PT'
    m = re.match(r'(@!?(?:P\d|PT))\s+', t)
    if m: g = m.group(1); t = t[m.end():]
    head, _, rest = t.partition(' ')
    glued = '[' in head  # CCTL.QRY1[R0+0x40]: the first operand follows the mnemonic unspaced
    if glued:
        k = head.index('['); rest = head[k:] + (', ' + rest if rest else ''); head = head[:k]
    parts = head.split('.')
    ops = [g] + split_ops(rest)
    shapes, atoms = [], []
    for j, o in enumerate(ops):
        if j: o = NONFIN.sub(r'\1', canon_brackets(o, zero))
        at = []
        def sub(mm):
            k, v = atomval(mm.group(1)); at.append((k, v, mm.group(1)))
            return {'R': PR, 'P': PP}.get(k, PN)
        shapes.append(ATOM.sub(sub, o)); atoms.append(at)
    return dict(base=parts[0], mods=tuple(parts[1:]), shapes=tuple(shapes), atoms=atoms, glued=glued)

FW = {16: 'e', 32: 'f', 64: 'd'}  # float widths nvdisasm prints
FM = {16: 10, 32: 23, 64: 52}  # their mantissa widths
NONFIN = re.compile(r'([+-](?:INF|[QS]NAN)|-0\.0) ')  # nvdisasm pads these: '+QNAN .NEG', '-0.0 , R3'
def fdecode(bits, W): return struct.unpack('<' + FW[W], bits.to_bytes(W // 8, 'little'))[0]

def fmt_float(bits, W):
    """nvdisasm's text for a W-bit float: +INF/-INF, +QNAN/-QNAN, -0.0, plain
    decimal (20 significant digits) from 1e-4 up to 1e9, else %.19e below
    one (trailing zeros cut) and %.20e above"""
    x = fdecode(bits, W); neg = bits >> (W - 1) & 1
    if math.isinf(x): return ('-' if neg else '+') + 'INF'
    if math.isnan(x): return ('-' if neg else '+') + fnan(bits, W)
    if x == 0: return '-0.0' if neg else '0'
    e = math.floor(math.log10(abs(x)))
    if -4 <= e < 9:
        t = format(Context(prec=20).plus(Decimal(x)), 'f')
        return t.rstrip('0').rstrip('.') if '.' in t else t
    if e < 0:
        m, ex = ('%.19e' % x).split('e')
        return m.rstrip('0').rstrip('.') + 'e' + ex
    return '%.20e' % x

def fnan(bits, W):
    """a NaN's name: halves and singles are printed widened to double, which makes them quiet"""
    return 'QNAN' if W < 64 or bits >> (FM[W] - 1) & 1 else 'SNAN'

def ftext_value(t):
    """a printed float's value (sign kept for zero, infinities and NaN), or None"""
    u = t.lstrip('+-'); sg = -1.0 if t.startswith('-') else 1.0
    if u == 'INF': return ('i', sg)
    if u in ('QNAN', 'SNAN'): return (u, sg)
    try: x = float(t)
    except ValueError: return None
    return ('v', x, math.copysign(1.0, x))

def fbits(enc, r):
    """the W-bit pattern an encoding 'W<<s' makes of field value r (s < 0: r's
    bits from -s up; a leading '-': printed negated, the sign bit flipped)"""
    W, s = map(int, enc.lstrip('-').split('<<'))
    b = (r << s if s >= 0 else r >> -s) & ((1 << W) - 1)
    return W, b ^ (1 << (W - 1)) if enc.startswith('-') else b

def fenc_text(enc, r): return fmt_float(*reversed(fbits(enc, r)))

def fmt_num(v, style):
    if style == 'hex': return ('-' if v < 0 else '') + hex(abs(v))
    return str(v)

def render(p, zero):
    """structured -> nvdisasm raw text"""
    ops = []
    for j, (s, at) in enumerate(zip(p['shapes'], p['atoms'])):
        it = iter(at); o = ''
        for ch in s:
            if ch in (PR, PP, PN):
                k, v, raw = next(it)
                if k == 'R': o += 'RZ' if v == 255 else f'R{v}'
                elif k == 'P': o += 'PT' if v == 7 else f'P{v}'
                elif raw == 'PC': o += 'PC' + ('-' if v < 0 else '+') + hex(abs(v))
                elif k == 'N': o += fmt_num(v, 'hex' if raw.lstrip('-').startswith('0x') else 'dec')
                else: o += raw + (' ' if NONFIN.match(raw + ' ') else '')
            else: o += ch
        o = re.sub(r'(?<!c)(?<!\])\[RZ\+0x0\]', '[' + (sorted(zero)[0] if zero else 'RZ') + ']', o) if j else o
        o = re.sub(r'\[RZ\+(-?0x[0-9a-f]+)\]', r'[\1]', o)
        o = re.sub(r'\[(R\d+)\+0x0\]', r'[\1]', o)
        ops.append(o)
    head = '.'.join((p['base'],) + tuple(p['mods']))
    g = '' if ops[0] == '@PT' else ops[0] + ' '
    if p.get('glued') and len(ops) > 1: return (g + head + ops[1] + (' ' + ', '.join(ops[2:]) if len(ops) > 2 else '')).rstrip()
    return (g + head + (' ' + ', '.join(ops[1:]) if len(ops) > 1 else '')).rstrip()

def shape_key(p):
    return (p['base'], p['mods'], p['shapes'])

def kinds(at): return tuple('N' if a[0] == 'F' else a[0] for a in at)

def strip_flags(s): return re.sub(r'[-|!~]', '', s)

def diff(b, v):
    """classes of one flip: list of tuples, first = primary"""
    if v is None: return [('reject',)]
    if v['base'] != b['base']: return [('opcode', f"{b['base']}>{v['base']}")]
    out = []
    if len(v['shapes']) != len(b['shapes']): return [('form',)]
    for j, (sb, sv) in enumerate(zip(b['shapes'], v['shapes'])):
        if sb != sv:
            if len(b['atoms'][j]) != len(v['atoms'][j]) or kinds(b['atoms'][j]) != kinds(v['atoms'][j]): return [('form',)]
            out.append(('flag' if strip_flags(sb) == strip_flags(sv) else 'enum', j, sb, sv))
    d = frozenset(((C(b['mods']) - C(v['mods'])) + (C(v['mods']) - C(b['mods']))).keys())  # a name printed twice (TLD4.B.B) counts twice
    if d: out.append(('mod', d))
    for j, (ab, av) in enumerate(zip(b['atoms'], v['atoms'])):
        if b['shapes'][j] != v['shapes'][j]: continue
        for n, (x, y) in enumerate(zip(ab, av)):
            if x[1] == y[1] and x[2] == y[2]: continue
            slot = f'g' if j == 0 else f'{j}.{n}'
            if x[0] in 'RP' and y[0] == x[0]:
                e = x[1] ^ y[1]
                out.append(('reg', slot, x[0], e.bit_length() - 1 if e & (e - 1) == 0 else None))
            elif x[0] == 'N' and y[0] == 'N':
                if (x[1] < 0) != (y[1] < 0): out.append(('imm', slot, 's'))
                else:
                    e = x[1] ^ y[1]
                    if e & (e - 1): e = abs(y[1] - x[1])  # a carry from a constant added to the value
                    out.append(('imm', slot, e.bit_length() - 1 if e & (e - 1) == 0 else None))
            else: out.append(('imm', slot, 'f'))
    out.sort(key=lambda c: ORDER.index(c[0]))
    return out or [('none',)]

# ---------------------------------------------------------------- pick
def pick():
    data, oprun = sys.argv[3], sys.argv[4]
    rows, found = {}, []
    # the manifest's rows, then the rows opcode-space sweeps found outside
    # them (new_rows.py), when that file exists
    for path in (data + '/decoder-rows.tsv', data + '/decoder-rows-discovered.tsv'):
        if not os.path.exists(path): continue
        for l in open(path):
            if l.startswith('#'): continue
            f = l.rstrip('\n').split('\t')
            if f[1] in ('root', 'discovered') and f[2] != 'OOPS': rows.setdefault(f[0], (f[2], int(f[3], 16), int(f[4], 16)))
            if f[1] == 'discovered': found.append((f[0], int(f[3], 16), int(f[4], 16)))
    sel = os.environ.get('INFER_ROWS', 'all')  # family-infer.sh --rows
    if sel != 'all': rows = {k: v for k, v in rows.items() if k in sel.split(',')}
    cand = []  # (row, word, source)
    for l in open(R + 'samples.words.tsv'):
        f = l.split('\t'); cand.append((f[0], int(f[2], 16), 'sample'))
    nv = load_nv('samples.nv.tsv')
    if oprun and os.path.exists(oprun + '/words.tsv'):
        for l in open(oprun + '/nv.tsv'):
            w, _, t = l.rstrip('\n').partition('\t')
            if not t.startswith('!'): nv[int(w, 16)] = t
        # a word no manifest row took goes to the first discovered row it
        # matches (runs made before those rows existed tag it '-')
        for l in open(oprun + '/words.tsv'):
            f = l.split('\t'); w = int(f[2], 16)
            if f[1] == '-': f[1] = next((k for k, v, m in found if w & m == v), '-')
            if f[1] in rows: cand.append((f[1], w, 'opcode-run'))
    seen, bases = set(), []
    for row, w, src in cand:
        p = parse(nv.get(w, '!'))
        if p is None: continue
        k = (row,) + shape_key(p)
        if k in seen: continue
        seen.add(k); bases.append((row, w, src, nv[w]))
    with open(R + 'bases.tsv', 'w') as o:
        for b in bases: o.write('%s\t%016x\t%s\t%s\n' % b)
    known = set(nv)
    flips = {}
    for row, w, _, _ in bases:
        for i in range(64):
            v = w ^ (1 << i)
            if v not in known: flips.setdefault(v, row)
    with open(R + 'flips.words.tsv', 'w') as o:
        for v, row in flips.items(): o.write(f'{row}\t{row}\t{v:016x}\tf\n')
    keep = {b[1] for b in bases} | {b[1] ^ (1 << i) for b in bases for i in range(64)}
    json.dump({k: v for k, v in nv.items() if k in keep}, open(R + 'known.json', 'w'))
    print(f'pick: {len(rows)} rows, {len({b[0] for b in bases})} with bases, {len(bases)} bases, {len(flips)} flip words', file=sys.stderr)

def all_nv():
    nv = {int(k): v for k, v in json.load(open(R + 'known.json')).items()}
    nv.update(load_nv('samples.nv.tsv', 'flips.nv.tsv', 'pairs.nv.tsv', 'tables.nv.tsv', 'validate.nv.tsv', 'pc1.nv.tsv', 'pc2.nv.tsv'))
    return nv

# ---------------------------------------------------------------- pc probe
# A number that names a branch target is the word's own address plus a
# value. Every base is asked again at a known address; a base whose text
# moved names its row and the atoms that moved, and that row's flips and
# pairs are asked again with addresses too.
def pc1():
    with open(R + 'pc1.words.tsv', 'w') as o:
        for row, w, _, _ in bases_iter(): o.write(f'{row}\t{row}\t{w:016x}\ta\n')

def pc2():
    old = {int(k): v for k, v in json.load(open(R + 'known.json')).items()}
    old.update(load_nv('samples.nv.tsv'))
    new = load_nv('pc1.nv.tsv'); rel = collections.defaultdict(set)
    for row, w, _, t in bases_iter():
        a, b = parse(old.get(w, t)), parse(new.get(w, '!'))
        if a is None or b is None or a['shapes'] != b['shapes']: continue
        for j, (x, y) in enumerate(zip(a['atoms'], b['atoms'])):
            for n, (u, v) in enumerate(zip(x, y)):
                if u[0] == v[0] == 'N' and u[1] != v[1]: rel[row].add(f'{j}.{n}')
    with open(R + 'pc.rel.tsv', 'w') as o:
        for row in sorted(rel, key=int):
            for a in sorted(rel[row]): o.write(f'{row}\t{a}\n')
    nv = all_nv(); out = {}
    for row, w, _, _ in bases_iter():
        if row not in rel: continue
        for x in [w ^ (1 << i) for i in range(64)] + [w ^ (3 << i) for i in range(63)]:
            if x in nv and not nv[x].startswith('!'): out.setdefault(x, row)
    with open(R + 'pc2.words.tsv', 'w') as o:
        for x, row in out.items(): o.write(f'{row}\t{row}\t{x:016x}\ta\n')
    print(f'pc: {len(rel)} rows with address-relative numbers ({sum(len(v) for v in rel.values())} atoms), {len(out)} words to ask again', file=sys.stderr)

def bases_iter():
    for l in open(R + 'bases.tsv'):
        f = l.rstrip('\n').split('\t'); yield f[0], int(f[1], 16), f[2], f[3]

# ---------------------------------------------------------------- pairs
def pairs():
    nv = all_nv(); out = {}
    for row, w, _, t in bases_iter():
        b = parse(t)
        roles = [diff(b, parse(nv.get(w ^ (1 << i), '!')))[0][0] for i in range(64)]
        for i in range(63):
            if roles[i] in IMPORTANT and roles[i + 1] in IMPORTANT:
                v = w ^ (3 << i)
                if v not in nv: out.setdefault(v, row)
    with open(R + 'pairs.words.tsv', 'w') as o:
        for v, row in out.items(): o.write(f'{row}\t{row}\t{v:016x}\tp\n')
    print(f'pairs: {len(out)} pair words', file=sys.stderr)

def joint(b, a, c, ab):
    """True when the pair flip ab is exactly flip a plus flip c"""
    if None in (a, c, ab): return False
    if a['base'] != b['base'] or c['base'] != b['base'] or ab['base'] != b['base']: return False
    if set(ab['mods']) != set(b['mods']) ^ (set(b['mods']) ^ set(a['mods'])) ^ (set(b['mods']) ^ set(c['mods'])): return False
    if not (len(a['shapes']) == len(c['shapes']) == len(ab['shapes']) == len(b['shapes'])): return False
    for j in range(len(b['shapes'])):
        ca = (a['shapes'][j], a['atoms'][j]) != (b['shapes'][j], b['atoms'][j])
        cc = (c['shapes'][j], c['atoms'][j]) != (b['shapes'][j], b['atoms'][j])
        if ca and cc: return False
        src = a if ca else c if cc else b
        if (ab['shapes'][j], ab['atoms'][j]) != (src['shapes'][j], src['atoms'][j]): return False
    return True

def dropped(full, short):
    """the operand index j whose removal from full leaves short (one atom, else None)"""
    if None in (full, short) or full['base'] != short['base'] or full['mods'] != short['mods']: return None
    if len(full['shapes']) != len(short['shapes']) + 1: return None
    for j in range(1, len(full['shapes'])):
        if len(full['atoms'][j]) == 1 and full['shapes'][:j] + full['shapes'][j + 1:] == short['shapes'] \
                and [[a[:2] for a in at] for at in full['atoms'][:j] + full['atoms'][j + 1:]] == [[a[:2] for a in at] for at in short['atoms']]:
            return j
    return None

def omissions(ev, texts_r):
    """operands printed only when their value is not one constant (PT, RZ, 0x0):
    ev holds (operand index, shape, operand count, the atom's printed value) for
    every flip that adds or drops that operand; the dropped value is the printed
    one with a single bit changed, the same for every flip and never printed"""
    cand, seen = {}, collections.defaultdict(set)
    for k, a in ev:
        x = {a ^ (1 << i) for i in range(64)}
        cand[k] = cand[k] & x if k in cand else x
    for _, p in texts_r:
        for k in cand:
            j, s_, n = k
            if len(p['shapes']) == n and p['shapes'][j] == s_: seen[k].add(p['atoms'][j][0][1])
    out = []
    for k, cs in cand.items():
        cs = cs - seen[k]
        if len(cs) == 1: out.append([*k, cs.pop()])
    return sorted(out)

def bits_str(lo, hi): return f'{hi}:{lo}' if hi != lo else f'{lo}'

# ---------------------------------------------------------------- layout
def pv(w, parts):
    """value of a field made of parts (low part first)"""
    v, s = 0, 0
    for lo, hi in parts:
        v |= ((w >> lo) & ((1 << (hi - lo + 1)) - 1)) << s; s += hi - lo + 1
    return v

def setpv(w, parts, v):
    for lo, hi in parts:
        n = hi - lo + 1; m = (1 << n) - 1
        w = (w & ~(m << lo)) | ((v & m) << lo); v >>= n
    return w

def pwidth(parts): return sum(hi - lo + 1 for lo, hi in parts)
def pstr(parts): return '+'.join(bits_str(lo, hi) for lo, hi in sorted(parts, reverse=True))

def classify_bits(row, gname, obs_g, X):
    """per-bit role over one form group's bases"""
    bits = []
    for i in range(64):
        ob = obs_g[i]
        inf = [c for _, c in ob if c[0][0] not in ('none', 'reject')]
        rc = C(c[0][0] for c in inf)
        nrej = sum(1 for _, c in ob if c[0][0] == 'reject'); nnone = sum(1 for _, c in ob if c[0][0] == 'none')
        role = rc.most_common(1)[0][0] if rc else ('opcode' if nrej else 'none')
        key, names = None, C()
        if role == 'reg':
            kc = C((c[0][1], c[0][2], i - c[0][3] if c[0][3] is not None else None) for c in inf if c[0][0] == 'reg')
            key = kc.most_common(1)[0][0]
            if len(kc) > 1: X.append((row, gname, i, 'slot-shift', ' '.join(f'{k}:{n}' for k, n in kc.items()), ob[0][0]))
        elif role == 'imm':
            kc = C(c[0][1] for c in inf if c[0][0] == 'imm'); key = (kc.most_common(1)[0][0],)
            if len(kc) > 1: X.append((row, gname, i, 'slot-shift', ' '.join(f'{k}:{n}' for k, n in kc.items()), ob[0][0]))
            vb = C(c[0][2] for c in inf if c[0][0] == 'imm' and c[0][1] == key[0]).most_common(1)[0][0]
            key = key + (vb,)
        elif role == 'mod':
            for c in inf:
                if c[0][0] == 'mod': names.update(c[0][1])
        elif role in ('enum', 'flag'):
            kc = C(c[0][1] for c in inf if c[0][0] == role); key = (kc.most_common(1)[0][0],)
        elif role == 'opcode':
            for c in inf:
                if c[0][0] == 'opcode': names[c[0][1]] += 1
            if nrej: names['invalid'] += nrej
        if len(rc) > 1:
            X.append((row, gname, i, 'multi-role', ' '.join(f'{k}:{n}' for k, n in rc.items()) + (f' reject:{nrej}' if nrej else ''),
                      next(w for w, c in ob if c[0][0] != role and c[0][0] not in ('none', 'reject'))))
        if rc and nnone:
            X.append((row, gname, i, 'conditional', f'{role}:{sum(rc.values())} none:{nnone}' + (f' reject:{nrej}' if nrej else ''),
                      next(w for w, c in ob if c[0][0] == 'none')))
        bits.append(dict(role=role, key=key, names=names))
    return bits

def runs(bits, depg):
    """contiguous runs of one role and key; selector bits join on pair evidence"""
    out, i = [], 0
    while i < 64:
        b = bits[i]; j = i
        while j + 1 < 64:
            nb = bits[j + 1]
            if nb['role'] != b['role']: break
            if b['role'] == 'reg' and nb['key'] != b['key']: break
            if b['role'] == 'imm' and nb['key'][0] != b['key'][0]: break
            if b['role'] in IMPORTANT:
                if b['role'] != 'mod' and nb['key'] != b['key']: break
                if depg[j][True] <= depg[j][False]: break
            j += 1
        out.append(dict(lo=i, hi=j, role=b['role'], bits=bits[i:j + 1]))
        i = j + 1
    return out

def amb(texts_g, parts, tokens, other=None):
    """modifier names a value of parts (plus other's value) leaves ambiguous"""
    tab = collections.defaultdict(C)
    for w, p in texts_g:
        k = pv(w, parts) if other is None else (pv(w, parts), pv(w, other))
        tab[k][tuple(t for t in p['mods'] if t in tokens)] += 1
    return sum(sum(c.values()) - c.most_common(1)[0][1] for c in tab.values()), tab

def cval(w, c, nz):
    """a key component's value in word w: its bits, or (nz) only whether they are all zero"""
    v = pv(w, c)
    return int(v != 0) if nz else v

def dependent(texts_g, comps, attr, vals=None, npair=None, nz=(), backoff=False):
    """the components whose joint value decides attr(w, p) over the words,
    grown one component at a time while that lowers the leave-one-out error
    (each word predicted by the other words of its key value; a key value
    seen once is predicted by what the other once-seen values show, which
    is also the rule's default for values never seen). A step may add two
    of the first npair components at once: a result decided by two bits
    together (one minus sign for either of two negations, none for both)
    shows nothing when either bit is tried alone. The table keeps only key
    values that name one result. Components whose index is in nz count
    only as zero or not zero (an immediate that must be zero under some
    selector values). backoff: a key value never seen takes the result its
    components name alone, where a component's value has only ever shown
    one result, else the result of the words none of whose component
    values decides alone (a refusal that follows either of two selectors)."""
    vals = {} if vals is None else vals
    obs = C()
    for w, p in texts_g:
        a = attr(w, p)
        if a is None: continue
        v = vals.get(w)
        if v is None: v = vals[w] = tuple(cval(w, c, k in nz) for k, c in enumerate(comps))
        obs[(v, a)] += 1
    if len({a for _, a in obs}) < 2: return None
    def tab(K):
        t = collections.defaultdict(C)
        for (v, a), m in obs.items(): t[tuple(v[k] for k in K)][a] += m
        return t
    def loo(t):
        err, once = 0, C()
        for c in t.values():
            n = sum(c.values())
            if n == 1: once.update(c); continue
            for a, m in c.items():
                if not all(m - 1 > x for b, x in c.items() if b != a): err += m
        for a, m in once.items():
            if not all(m - 1 > x for b, x in once.items() if b != a): err += m
        return err
    K, e0 = [], loo(tab([]))
    while e0:
        free = [k for k in range(len(comps)) if k not in K]
        steps = [(k,) for k in free] + list(itertools.combinations([k for k in free if k < (len(comps) if npair is None else npair)], 2))
        best = min(((loo(t), len(ks), len(t), ks) for ks in steps for t in [tab(K + list(ks))]), default=None)
        if best is None or best[0] >= e0: break
        e0 = best[0]; K += list(best[3])
    if not K: return None
    t = tab(K)
    once = C()
    for c in t.values():
        if sum(c.values()) == 1: once.update(c)
    top = once.most_common(2)
    default = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else None
    rule = dict(key=[comps[k] for k in K], default=default,
                table={','.join(map(str, v)): next(iter(c)) for v, c in t.items() if len(c) == 1})
    if any(k in nz for k in K): rule['nz'] = [i for i, k in enumerate(K) if k in nz]
    if backoff and len(K) > 1:
        marg = [collections.defaultdict(set) for _ in K]
        for (v, a), m in obs.items():
            for i, k in enumerate(K): marg[i][v[k]].add(a)
        pure = [{str(x): next(iter(rs)) for x, rs in mg.items() if len(rs) == 1} for mg in marg]
        rest = C()
        for (v, a), m in obs.items():
            if not any(str(v[k]) in pure[i] for i, k in enumerate(K)): rest[a] += m
        rule['pure'] = pure
        top = rest.most_common(2)
        if top and (len(top) == 1 or top[0][1] > top[1][1]): rule['rest'] = top[0][0]
    return rule

def keyed(rule, w):
    """a dependent rule's result for word w: its table, else (a backoff rule) what its
    components decide alone, else its default (None when it has none)"""
    if not rule: return None
    nz = rule.get('nz', ())
    v = [str(cval(w, c, i in nz)) for i, c in enumerate(rule['key'])]
    r = rule['table'].get(','.join(v))
    if r is not None or 'pure' not in rule: return rule.get('default') if r is None else r
    alone = {rule['pure'][i][x] for i, x in enumerate(v) if x in rule['pure'][i]}
    if len(alone) == 1: return alone.pop()
    return rule.get('rest', rule.get('default')) if not alone else rule.get('default')

def imm_value(f, w, twos):
    """f's value in word w from its bit weights; twos: the top bit counts negative"""
    v, ws = 0, f['weights']; k = 0
    for lo, hi in f['parts']:
        for i in range(lo, hi + 1):
            if (w >> i) & 1: v += -ws[k] if twos and k == len(ws) - 1 else ws[k]
            k += 1
    return v

def imm_attrs(f):
    """how words print f's value: sign ('s' two's complement, 'u' plain; known
    only when the top bit is set) and negation (printed as minus the value)"""
    j, n = (0, 0) if f['slot'] == 'g' else map(int, f['slot'].split('.'))
    top = f['parts'][-1][1]
    def got(p):
        if j >= len(p['atoms']) or n >= len(p['atoms'][j]) or p['atoms'][j][n][0] != 'N': return None
        return p['atoms'][j][n][1]
    b0 = f.get('bias', 0)
    def sign(w, p):
        g = got(p)
        if g is None or not (w >> top) & 1: return None
        g -= b0
        u, s_ = imm_value(f, w, False), imm_value(f, w, True)
        # the value as printed decides first; only else a negation may explain it
        k = {a for a, v in (('u', u), ('s', s_)) if g == v} or {a for a, v in (('u', u), ('s', s_)) if g == -v}
        return k.pop() if len(k) == 1 else None
    def neg(w, p):
        g = got(p)
        if g is None or g == b0: return None
        g -= b0
        vs = {imm_value(f, w, False), imm_value(f, w, True)}
        k = {a for a, x in ((0, g), (1, -g)) if x in vs}
        return k.pop() if len(k) == 1 else None
    return sign, neg

def build_group(row, gname, gbases, obs, depw, texts_g, texts_r, enums, X, refused=()):
    gb = set(gbases)
    obs_g = {i: [(w, c) for w, c in obs[i] if w in gb] for i in range(64)}
    depg = {i: C(d for w, d in depw[i] if w in gb) for i in range(64)}
    rr = runs(classify_bits(row, gname, obs_g, X), depg)
    fields, imms = [], collections.defaultdict(list)
    for f in rr:
        lo, hi, role = f['lo'], f['hi'], f['role']; fd = dict(parts=[[lo, hi]], comps=[[[lo, hi]]], role=role)
        if role == 'reg':
            slot, kind, off = f['bits'][0]['key']; fd.update(slot=slot, kind=kind, start_ok=off == lo)
        elif role == 'imm':
            imms[f['bits'][0]['key'][0]].append(f); continue
        elif role == 'mod':
            toks = set()
            for b in f['bits']: toks |= set(b['names'])
            fd['tokens'] = toks
        elif role in ('enum', 'flag'):
            fd['slot'] = f['bits'][0]['key'][0]
        elif role == 'opcode':
            nm = C()
            for b in f['bits']: nm.update(b['names'])
            fd['names'] = [k for k, _ in nm.most_common(6)]
        fields.append(fd)
    # immediates: every run of one operand slot is one value, low run first; a
    # top bit that flips the sign, or moves the value by more than one power of
    # two, makes it two's complement over the whole width
    for slot, fs in imms.items():
        fs.sort(key=lambda f: f['lo'])
        vbs = [b['key'][1] for f in fs for b in f['bits']]
        ints = [(k, vb) for k, vb in enumerate(vbs) if isinstance(vb, int)]
        shift = ints[0][1] - ints[0][0] if ints else None
        signed = vbs[-1] in ('s', None)
        body = vbs[:-1] if signed else vbs
        regular = shift is not None and all(vb - k == shift for k, vb in enumerate(body) if isinstance(vb, int)) \
            and all(isinstance(vb, int) for vb in body)
        # the bit list: each bit's weight (a sign bit weighs the next power up)
        hi_ = max((vb for vb in vbs if isinstance(vb, int)), default=-1)
        weights = [1 << vb if isinstance(vb, int) and vb >= 0 else 1 << (hi_ + 1) if vb in ('s', None) and k == len(vbs) - 1 else None
                   for k, vb in enumerate(vbs)]
        fields.append(dict(parts=[[f['lo'], f['hi']] for f in fs], role='imm', slot=slot, shift=shift if regular else None,
                           signed=signed, float='f' in vbs, regular=regular, weights=None if None in weights else weights))
    # modifier fields that print overlapping names are one field
    mods = [f for f in fields if f['role'] == 'mod']
    merged = True
    while merged:
        merged = False
        for a in mods:
            for b in mods:
                if a is not b and a['tokens'] & b['tokens']:
                    a['parts'] += b['parts']; a['comps'] += b['comps']; a['tokens'] |= b['tokens']; mods.remove(b); fields.remove(b); merged = True; break
            if merged: break
    # enumerated values add the names they print
    for f in mods:
        ew = enums.get(pstr(f['parts']))
        if ew:
            sets = [set(p['mods']) for _, p in ew]
            f['tokens'] |= set().union(*sets) - set.intersection(*sets)
    # a name that still depends on another selector field: join them
    for f in list(mods):
        if f not in mods: continue
        a0, _ = amb(texts_g, f['parts'], f['tokens'])
        while a0:
            best = None
            for g in fields:
                if g is f or g['role'] not in ('mod', 'enum', 'flag'): continue
                a1, _ = amb(texts_g, f['parts'], f['tokens'] | g.get('tokens', set()), g['parts'])
                if a1 < a0 and (best is None or a1 < best[0]): best = (a1, g)
            if best is None: break
            a0, g = best
            f['parts'] = f['parts'] + g['parts']; f['comps'] = f['comps'] + g['comps']; f.setdefault('joined', []).append(pstr(g['parts']))
            if g['role'] == 'mod': f['tokens'] |= g['tokens']; fields.remove(g); mods.remove(g)
    need = []
    for f in fields:
        f['parts'].sort(key=lambda p: p[0]) if f['role'] != 'imm' else None
        if f['role'] in ('mod', 'enum'):
            # every value of a single field; a joint field is the product of
            # its components' valid values, each component enumerated first
            if pstr(f['parts']) not in enums:
                if len(f['comps']) == 1: need.append(('all', f['parts'], None))
                else:
                    missing = [c for c in f['comps'] if pstr(c) not in enums]
                    if missing: need += [('all', c, None) for c in missing]
                    else: need.append(('product', f['parts'], [(c, sorted({pv(x, c) for x, _ in enums[pstr(c)]})) for c in f['comps']]))
            tab = collections.defaultdict(C)
            for w, p in texts_g:
                v = pv(w, f['parts'])
                if f['role'] == 'mod': tab[v][tuple(t for t in p['mods'] if t in f['tokens'])] += 1
                elif f['slot'] < len(p['shapes']): tab[v][p['shapes'][f['slot']]] += 1
            f['table'] = {str(v): list(c.most_common(1)[0][0]) if f['role'] == 'mod' else c.most_common(1)[0][0] for v, c in tab.items()}
            if f['role'] == 'mod':
                for v, c in tab.items():
                    if len(c) > 1:
                        X.append((row, gname, f['parts'][0][0], 'value-ambiguous', f'bits {pstr(f["parts"])} value {v}: ' +
                                  ' '.join(('.'.join(k) or '-') + f':{n}' for k, n in c.most_common()), '-'))
            if 'tokens' in f: f['tokens'] = sorted(f['tokens'])
    def slot_jn(sl): return (0, 0) if sl == 'g' else tuple(map(int, sl.split('.')))
    # a register field may print its value inverted (a predicate written as
    # PT for 0, P0 for 7): the printed number is the field's value xor a
    # constant, learned as the one most words show
    for f in fields:
        if f['role'] != 'reg': continue
        fj, fn = slot_jn(f['slot'])
        xc = C(p['atoms'][fj][fn][1] ^ pv(w, f['parts']) for w, p in texts_g
               if fj < len(p['atoms']) and fn < len(p['atoms'][fj]) and p['atoms'][fj][fn][0] == f['kind'])
        if xc and xc.most_common(1)[0][0]: f['xor'] = xc.most_common(1)[0][0]
    # an operand no field sets that always prints another register field's
    # value is that field printed twice (a destination read back as a source)
    claimed = {slot_jn(f['slot']) for f in fields if f['role'] in ('reg', 'imm')}
    p0 = texts_g[0][1] if texts_g else None
    for j, at in enumerate(p0['atoms'] if p0 else []):
        for n, a in enumerate(at):
            if (j, n) in claimed or a[0] not in 'RP': continue
            for f in fields:
                if f['role'] != 'reg' or f['kind'] != a[0]: continue
                fj, fn = slot_jn(f['slot']); seen, same = set(), True
                for w, p in texts_g:
                    if j < len(p['atoms']) and n < len(p['atoms'][j]) and fj < len(p['atoms']) and fn < len(p['atoms'][fj]):
                        if p['atoms'][j][n] [1] != p['atoms'][fj][fn][1]: same = False; break
                        seen.add(p['atoms'][j][n][1])
                if same and len(seen) > 1: f.setdefault('also', []).append(f'{j}.{n}'); break
    # a register operand may print another register field's value plus a
    # constant (the second of a register pair), or the zero register, as
    # selector values decide: per register atom and source field, the rule
    # that reproduces every word of the form (an atom the field already
    # prints as itself needs none)
    sel0 = list({pstr(c): c for f in fields if f['role'] in ('mod', 'enum', 'flag') for c in f['comps']}.values())
    regf = [f for f in fields if f['role'] == 'reg']
    p0 = texts_g[0][1] if texts_g else None
    for j, at in enumerate(p0['atoms'] if p0 else []):
        for n, a in enumerate(at):
            if a[0] not in 'RP': continue
            zr = 255 if a[0] == 'R' else 7
            best = None
            for f in [f for f in regf if f['kind'] == a[0] and f['start_ok'] and pwidth(f['parts']) > 1]:
                def rel(w, p, f=f, j=j, n=n, zr=zr):
                    if j >= len(p['atoms']) or n >= len(p['atoms'][j]) or p['atoms'][j][n][0] != a[0]: return None
                    fv = pv(w, f['parts']) ^ f.get('xor', 0)
                    if fv == zr: return None
                    v = p['atoms'][j][n][1]
                    return 'Z' if v == zr else v - fv
                got = [(w, r) for w, p in texts_g for r in [rel(w, p)] if r is not None]
                if not got or len({p['atoms'][j][n][1] for w, p in texts_g if j < len(p['atoms']) and n < len(p['atoms'][j])}) < 2: continue
                # keys: selectors, and one-bit register "fields" (a flip there moved this atom: a selector of it)
                ks = sel0 + [g['parts'] for g in regf if pwidth(g['parts']) == 1]
                r = dependent(texts_g, ks, rel, {})
                if r is None:
                    rs = {x for _, x in got}
                    if len(rs) != 1 or rs == {0} or rs == {'Z'}: continue
                    r = dict(key=[], default=rs.pop(), table={})
                # the field the most distinct values show it follows
                if all(keyed(r, w) == x for w, x in got):
                    nf = len({pv(w, f['parts']) for w, _ in got})
                    if best is None or nf > best[0]: best = (nf, f, r)
            if best: best[1].setdefault('derive', []).append([j, n, best[2]])
    # an operand's rendering (its shape, an immediate's signedness) may be
    # decided by selector values, alone or in combination: learn which
    # (selectors and register fields are the candidate keys; on equal error
    # the key with fewer values wins).
    # A shape is keyed by selectors only and applies only on key values seen.
    sel = list({pstr(c): c for f in fields if f['role'] in ('mod', 'enum', 'flag') for c in f['comps']}.values())
    comps = sel + list({pstr(f['parts']): f['parts'] for f in fields if f['role'] == 'reg'}.values())
    render, gv, rv = {}, {}, {}
    nslots = max((len(p['shapes']) for _, p in texts_g), default=0)
    for j in range(nslots):
        r = dependent(texts_g, sel, lambda w, p: p['shapes'][j] if j < len(p['shapes']) else None, gv)
        if r: r['default'] = None; render[str(j)] = r
    # a modifier no field owns may still follow selector values together
    # (IADD/IMAD print .PO for both negations set, and no minus signs)
    owned = {t for f in fields if f['role'] == 'mod' for t in f['tokens']}
    for t in sorted({t for _, p in texts_g for t in p['mods']} - owned):
        r = dependent(texts_g, sel, lambda w, p: int(t in p['mods']), gv)
        # only when the key decides it on every value seen: a value showing
        # both means the token follows something the selectors do not hold
        if r and len(r['table']) == len({','.join(str(pv(w, c)) for c in r['key']) for w, _ in texts_g}):
            r['default'] = None; render['mod:' + t] = r
    # bits with no effect on this form's bases may be a number other
    # selector values print (a bound texture's index, printed 0x0 when
    # bindless): a number operand no immediate sets that equals the bits'
    # value on every word printing it nonzero is that immediate
    nclaimed = {slot_jn(f['slot']) for f in fields if f['role'] == 'imm'}
    for f in [f for f in fields if f['role'] == 'none']:
        for j, at in enumerate(p0['atoms'] if p0 else []):
            for n, a in enumerate(at):
                if a[0] != 'N' or (j, n) in nclaimed: continue
                got = [(p['atoms'][j][n][1], pv(w, f['parts'])) for w, p in texts_g
                       if j < len(p['atoms']) and n < len(p['atoms'][j]) and p['atoms'][j][n][0] == 'N' and p['atoms'][j][n][1]]
                # two set bits that land where the field puts them, at least, so that one
                # word whose operand happens to equal the bits does not make an immediate
                if got and all(g == v for g, v in got) and bin(sum(1 << i for g, _ in got for i in range(64) if g >> i & 1)).count('1') > 1:
                    f.update(role='imm', slot=f'{j}.{n}', shift=0, signed=False, float=False, regular=True,
                             weights=[1 << k for k in range(pwidth(f['parts']))])
                    nclaimed.add((j, n)); break
            if f['role'] == 'imm': break
    for f in fields:
        if f['role'] == 'imm' and f['weights'] and not f['float']:
            # a constant the printed number adds (a branch target counts from the next word)
            j, n = (0, 0) if f['slot'] == 'g' else map(int, f['slot'].split('.'))
            top = f['parts'][-1][1]
            # (a zero printed for a nonzero value is a number not printed, not a constant)
            bc = C(p['atoms'][j][n][1] - imm_value(f, w, False) for w, p in texts_g
                   if not (w >> top) & 1 and j < len(p['atoms']) and n < len(p['atoms'][j]) and p['atoms'][j][n][0] == 'N'
                   and (p['atoms'][j][n][1] or not imm_value(f, w, False)))
            if bc and bc.most_common(1)[0][0]: f['bias'] = bc.most_common(1)[0][0]
            sign, neg = imm_attrs(f)
            own = {pstr(c) for c in [f['parts']]}
            ks = [c for c in comps if pstr(c) not in own]
            np_ = len([c for c in sel if pstr(c) not in own])
            r = dependent(texts_g, ks, sign, rv, np_)
            if r: f['sign_by'] = r
            r = dependent(texts_g, ks, neg, rv, np_)
            if r: f['neg_by'] = r
            # a value printed as zero under some selector values
            def zero(w, p, f=f, j=j, n=n):
                if j >= len(p['atoms']) or n >= len(p['atoms'][j]) or p['atoms'][j][n][0] != 'N': return None
                g, u, s_ = p['atoms'][j][n][1], imm_value(f, w, False), imm_value(f, w, True)
                if g == 0: return 'z' if u and s_ else None
                return 'v' if g - f.get('bias', 0) in (u, s_, -u, -s_) else None
            r = dependent(texts_g, sel, zero, gv)
            if r is None:
                # too few words for leave-one-out (a form of one base): a selector
                # whose every value shows one result decides it
                zs = [(w, a) for w, p in texts_g for a in [zero(w, p)] if a]
                for c in sel:
                    t = collections.defaultdict(set)
                    for w, a in zs: t[str(pv(w, c))].add(a)
                    if len(t) > 1 and all(len(v) == 1 for v in t.values()) and len({next(iter(v)) for v in t.values()}) > 1:
                        r = dict(key=[c], default=None, table={k: next(iter(v)) for k, v in t.items()}); break
            if r and 'z' in r['table'].values(): f['zero_by'] = r
        elif f['role'] == 'imm' and f['float']:
            # a float immediate: its bits are the top (or some run) of a 16,
            # 32 or 64-bit IEEE number; which one may follow selector values
            j, n = slot_jn(f['slot']); nb = pwidth(f['parts'])
            cands = [f'{m}{W}<<{s_}' for m in ('', '-') for W in FW for s_ in range(min(0, W - nb), max(0, W - nb) + 1)]
            def enc(w, p, j=j, n=n, f=f, cands=cands):
                if j >= len(p['atoms']) or n >= len(p['atoms'][j]) or p['atoms'][j][n][0] not in 'NF': return None
                tv = ftext_value(p['atoms'][j][n][2]); r = pv(w, f['parts'])
                if tv is None: return None
                k = set()
                for e_ in cands:
                    W, b_ = fbits(e_, r); x = fdecode(b_, W); neg = -1.0 if b_ >> (W - 1) & 1 else 1.0
                    got = ('i', neg) if math.isinf(x) else (fnan(b_, W), neg) if math.isnan(x) \
                        else ('v', x, math.copysign(1.0, x))
                    if got == tv: k.add(e_)
                return k.pop() if len(k) == 1 else None
            own = pstr(f['parts'])
            ks = [c for c in sel if pstr(c) != own]
            ec = C(e for w, p in texts_g for e in [enc(w, p)] if e)
            if ec: f['fenc'] = ec.most_common(1)[0][0]
            r = dependent(texts_g, ks, enc, {})
            if r: f['enc_by'] = r
    # invalid value combinations: words nvdisasm refused that differ from
    # their base only in this form's value fields, against every accepted
    # word, keyed by selectors and single immediate bits
    vbits = 0
    for f in fields:
        if f['role'] not in ('opcode', 'form'):
            for lo, hi in f['parts']: vbits |= ((1 << (hi - lo + 1)) - 1) << lo
    bad = [(x, 'bad') for b_, x in refused if not (x ^ b_) & ~vbits]
    if bad:
        # candidate keys: selectors, whether an immediate is zero, and its single bits
        ivals = [f['parts'] for f in fields if f['role'] == 'imm']
        ibits = [[[i, i]] for f in fields if f['role'] == 'imm' for lo, hi in f['parts'] for i in range(lo, hi + 1)]
        r = dependent([(w, 'ok') for w, _ in texts_g] + bad, sel + ivals + ibits, lambda w, a: a, None, len(sel) + len(ivals),
                      set(range(len(sel), len(sel) + len(ivals))), True)
        if r: render['valid'] = r
    return fields, need, render

def field_row(row, gname, f):
    r = f['role']; parts = f['parts']; name, desc = r, ''
    if r == 'reg':
        name = f"{f['kind']}@{f['slot']}"
        desc = f"operand {f['slot']} {f['kind']}, width {pwidth(parts)}" + ('' if f['start_ok'] else ', start mismatch') + \
               (f", also prints as {' '.join(f['also'])}" if f.get('also') else '')
        if f.get('xor'): desc += f", printed xor {f['xor']}"
        for j, n, rule in f.get('derive', []):
            desc += f", prints {j}.{n} as itself plus " + ('|'.join(pstr(c) for c in rule['key']) + ' ' if rule['key'] else '') + \
                ';'.join(f'{v}={t}' for v, t in sorted(rule['table'].items())) + (f" else {rule['default']}" if rule.get('default') is not None else '')
    elif r == 'imm':
        name = f"imm@{f['slot']}"
        desc = (f"scale {1 << f['shift']}" if f['shift'] is not None and f['shift'] >= 0 else 'scale ?') + \
               (' signed' if f['signed'] else '') + (' float' if f['float'] else '') + ('' if f['regular'] else ' irregular') + \
               ('' if f.get('weights') else ' no value') + (f" plus {f['bias']}" if f.get('bias') else '')
        if f.get('fenc'): desc += f", {f['fenc']}"
        for nm in ('sign_by', 'neg_by', 'enc_by', 'zero_by'):
            if f.get(nm):
                sb = f[nm]
                desc += f', {nm[:-3]} by ' + '|'.join(pstr(c) for c in sb['key']) + ' ' + ';'.join(f'{v}={t}' for v, t in sorted(sb['table'].items())[:24]) + \
                    (f" else {sb['default']}" if sb.get('default') is not None else '')
    elif r == 'mod':
        desc = ';'.join(f'{v}=' + ('.'.join(f['table'][v]) or '-') for v in sorted(f['table'], key=int))
        if f.get('joined'): name = 'mod(joint)'
    elif r == 'enum':
        name = f"enum@{f['slot'] or 'g'}"; desc = ';'.join(f"{v}={disp(f['table'][v])}" for v in sorted(f['table'], key=int))
    elif r == 'flag':
        name = f"flag@{f['slot'] or 'g'}"; desc = f"operand {f['slot'] or 'g'}"
    elif r == 'opcode':
        desc = ' '.join(f['names'])
    return (row, gname, name, pstr(parts), r, desc)

def layout():
    nv = all_nv(); rel = load_rel()
    obs = collections.defaultdict(lambda: collections.defaultdict(list))   # row -> bit -> [(base, classes)]
    depw = collections.defaultdict(lambda: collections.defaultdict(list))  # row -> bit -> [(base, joined?)]
    texts = collections.defaultdict(list)  # row -> parsed accepted words (w, p)
    zero = collections.defaultdict(set)
    trans = collections.defaultdict(dict)  # row -> 'slot|shape|bit' -> shape
    bases = collections.defaultdict(list); bparse = {}
    rejects = collections.defaultdict(list)  # row -> (base, word nvdisasm refused)
    omit = collections.defaultdict(list)  # row -> (operand, printed value) of flips that add or drop it
    def refused(x): return x in nv and (nv[x].startswith('!REJECT') or 'INVALID' in nv[x])
    for row, w, src, t in bases_iter():
        z = zero[row]; b = at_pc(parse(nv.get(w, t), z), w, rel[row])
        if b is None: continue
        texts[row].append((w, b)); bases[row].append(w); bparse[w] = b
        ps = []
        for i in range(64):
            v = at_pc(parse(nv.get(w ^ (1 << i), '!'), z), w ^ (1 << i), rel[row]); ps.append(v)
            if refused(w ^ (1 << i)): rejects[row].append((w, w ^ (1 << i)))
            c = diff(b, v); obs[row][i].append((w, c))
            if v and c[0][0] in ('opcode', 'form'): ps[-1] = None
            if c[0][0] == 'flag' and len(c) == 1: trans[row][f'{c[0][1]}|{c[0][2]}|{i}'] = c[0][3]
            if ps[-1]: texts[row].append((w ^ (1 << i), v))
            for full, short in ((b, v), (v, b)):
                j = dropped(full, short)
                if j is not None and full['atoms'][j][0][0] in 'RPN':
                    omit[row].append(((j, full['shapes'][j], len(full['shapes'])), full['atoms'][j][0][1]))
        for i in range(63):
            if obs[row][i][-1][1][0][0] in IMPORTANT and obs[row][i + 1][-1][1][0][0] in IMPORTANT:
                ab = at_pc(parse(nv.get(w ^ (3 << i), '!'), z), w ^ (3 << i), rel[row])
                if refused(w ^ (3 << i)): rejects[row].append((w, w ^ (3 << i)))
                if ab and ab['base'] == b['base'] and len(ab['shapes']) == len(b['shapes']): texts[row].append((w ^ (3 << i), ab))
                depw[row][i].append((w, not joint(b, ps[i], ps[i + 1], ab)))
    # enumerations from earlier passes: tag e|group|parts|base
    enums = collections.defaultdict(lambda: collections.defaultdict(list))  # (row, group) -> parts -> [p]
    if os.path.exists(R + 'tables.words.tsv'):
        for l in open(R + 'tables.words.tsv'):
            row, _, x, tag = l.rstrip('\n').split('\t')
            _, g, ps_, bw = tag.split('|')
            b = bparse.get(int(bw, 16)); p = at_pc(parse(nv.get(int(x, 16), '!'), zero[row]), int(x, 16), rel[row])
            enums[(row, g)].setdefault(ps_, [])
            if b is not None and refused(int(x, 16)): rejects[row].append((int(bw, 16), int(x, 16)))
            if b is None or p is None or p['base'] != b['base'] or len(p['shapes']) != len(b['shapes']): continue
            texts[row].append((int(x, 16), p)); enums[(row, g)][ps_].append((int(x, 16), p))
    model, L, X, new = {}, [], [], []
    # modifier order over every row, per mnemonic: what a row never printed together
    mbefore = collections.defaultdict(C)
    for row in texts:
        for w, p in texts[row]:
            m = p['mods']
            for a in range(len(m)):
                for c in range(a + 1, len(m)): mbefore[p['base']][(m[a], m[c])] += 1
    for row in sorted(obs, key=int):
        # operand form: the bits whose flip changes the operand list select it
        fmask = 0
        for i in range(64):
            if any(c[0][0] == 'form' for _, c in obs[row][i]): fmask |= 1 << i
        groups = collections.defaultdict(list)
        for w in bases[row]: groups[w & fmask].append(w)
        gtexts = collections.defaultdict(list)
        for w, p in texts[row]: gtexts[w & fmask].append((w, p))
        M = dict(fmask=fmask, groups={}, trans=trans[row], zero=sorted(zero[row]), pcrel=sorted(rel[row]),
                 omit=omissions(omit[row], texts[row]))
        before = C()
        for w, p in texts[row]:
            m = p['mods']
            for a in range(len(m)):
                for c in range(a + 1, len(m)): before[(m[a], m[c])] += 1
        M['before'] = [[a, b, n] for (a, b), n in before.items()]
        toks = {t for _, p in texts[row] for t in p['mods']}
        M['before_m'] = sorted([a, b, n] for mn in {p['base'] for _, p in texts[row]} for (a, b), n in mbefore[mn].items()
                               if a in toks and b in toks and not before[(a, b)] and not before[(b, a)])
        for g, gb in sorted(groups.items()):
            gname = f'{g:x}' if fmask else '-'
            fields, need, render = build_group(row, gname, gb, obs[row], depw[row], gtexts[g], texts[row], enums[(row, gname)], X,
                                                [(b_, x) for b_, x in rejects[row] if b_ & fmask == g and x & fmask == g])
            M['groups'][gname] = dict(fields=fields, bases=['%016x' % w for w in gb], render=render)
            for f in fields: L.append(field_row(row, gname, f))
            for j, r in render.items():
                L.append((row, gname, f'render@{j}', '|'.join(pstr(c) for c in r['key']), 'render',
                          ';'.join(f'{v}={disp(str(t))}' for v, t in sorted(r['table'].items())[:32]) + (f" else {disp(r['default'])}" if r.get('default') else '')))
            b0 = gb[0]
            for kind, parts, comps in need:
                tag = f'e|{gname}|{pstr(parts)}|{b0:016x}'
                if kind == 'all':
                    for v in range(1 << pwidth(parts)): new.append((row, setpv(b0, parts, v), tag))
                else:
                    for vals in itertools.product(*[vs for _, vs in comps]):
                        x = b0
                        for (c, _), v in zip(comps, vals): x = setpv(x, c, v)
                        new.append((row, x, tag))
        # an immediate whose form shows one sign only (an ATOM.CAS form whose
        # words all address through RZ) learns it over the forms holding the
        # same field, keyed by the register fields they share
        sib = collections.defaultdict(list)
        for g in groups:
            gname = f'{g:x}' if fmask else '-'
            for f in M['groups'][gname]['fields']:
                if f['role'] == 'imm' and f['weights'] and not f['float']: sib[(f['slot'], pstr(f['parts']))].append((g, f))
        for (slot, ps_), lst in sib.items():
            if len(lst) < 2: continue
            regs = None
            for g, _ in lst:
                rp = {pstr(h['parts']): h['parts'] for h in M['groups'][f'{g:x}' if fmask else '-']['fields'] if h['role'] == 'reg'}
                regs = rp if regs is None else {k: v for k, v in regs.items() if k in rp}
            pool = [x for g, _ in lst for x in gtexts[g]]
            for g, f in lst:
                sign, _ = imm_attrs(f)
                if f.get('sign_by') or len({sign(w, p) for w, p in gtexts[g]} - {None}) > 1: continue
                r = dependent(pool, list(regs.values()), sign, {})
                if r:
                    f['sign_by'] = r
                    L.append((row, f'{g:x}' if fmask else '-', f'sign@{slot}', '|'.join(pstr(c) for c in r['key']), 'render',
                              ';'.join(f'{v}={t}' for v, t in sorted(r['table'].items())[:24]) + (f" else {r['default']}" if r.get('default') else '') + ' (from forms sharing it)'))
        model[row] = M
    with open(R + 'layouts.tsv', 'w') as o:
        o.write('#row\tform\tfield\tbits\trole\tnames\n')
        for t in L: o.write('\t'.join(map(str, t)) + '\n')
    with open(R + 'conflicts.tsv', 'w') as o:
        o.write('#row\tform\tbit\tkind\tcounts\texample_base\n')
        for t in X: o.write('\t'.join(map(str, t[:5])) + '\t' + (t[5] if isinstance(t[5], str) else '%016x' % t[5]) + '\n')
    json.dump(model, open(R + 'model.json', 'w'))
    with open(R + 'tables.new.words.tsv', 'w') as o, open(R + 'tables.ask.words.tsv', 'w') as a:
        asked = set()
        for row, x, tag in new:
            o.write(f'{row}\t{row}\t{x:016x}\t{tag}\n')
            if (x not in nv or (rel[row] and x not in ADDR and not nv[x].startswith('!'))) and x not in asked: asked.add(x); a.write(f'{row}\t{row}\t{x:016x}\te\n')
    ng = sum(len(m['groups']) for m in model.values())
    print(f'layout: {len(model)} rows, {ng} form groups, {len(L)} fields, {len(X)} conflicts '
          f'({", ".join(f"{k} {n}" for k, n in C(x[3] for x in X).items())}), {len(new)} enumeration words to ask', file=sys.stderr)

# ---------------------------------------------------------------- compose / validate
PER_ROW = int(os.environ.get('INFER_PER_ROW', '16'))

def order_mods(chains, before, other=None):
    """merge suffix chains: each chain keeps its own order (a modifier
    field's value prints its names in the order seen for that value); the
    row's pairwise counts order the chains' heads against each other, and
    a pair the row never printed together takes the counts of the rows
    printing the same mnemonic (other)"""
    chains, out = [list(c) for c in chains if c], []
    def cnt(x, y):
        if before.get((x, y), 0) or before.get((y, x), 0) or not other: return before.get((x, y), 0)
        return other.get((x, y), 0)
    while chains:
        heads = [c[0] for c in chains]
        k = next((i for i, x in enumerate(heads) if all(cnt(y, x) <= cnt(x, y) for y in heads if y != x)), 0)
        out.append(chains[k].pop(0)); chains = [c for c in chains if c]
    return tuple(out)

def predict(model, G, base_w, b, new_w):
    """structured prediction for new_w composed on base_w, or None"""
    fields = G['fields']
    p = dict(base=b['base'], glued=b['glued'], mods=list(b['mods']), shapes=list(b['shapes']), atoms=[list(a) for a in b['atoms']])
    before = {(a, c): n for a, c, n in model['before']}
    owned = {t for f in fields if f['role'] == 'mod' for t in f['tokens']}
    chains = [[t for t in b['mods'] if t not in owned]]
    for f in fields:
        if f['role'] == 'mod':
            v = str(pv(new_w, f['parts']))
            if pv(new_w, f['parts']) != pv(base_w, f['parts']):
                if v not in f['table']: return None
                chains.append(f['table'][v])
            else: chains.append([t for t in b['mods'] if t in f['tokens']])
        parts = f['parts']; w = pwidth(parts)
        nv_, ov = pv(new_w, parts), pv(base_w, parts)
        if f['role'] == 'imm' and f['float']:
            # printed from its bits in the encoding the selectors name
            e = keyed(f.get('enc_by'), new_w) or f.get('fenc')
            if e is None:
                if nv_ != ov: return None
                continue
            j, n = (0, 0) if f['slot'] == 'g' else map(int, f['slot'].split('.'))
            if j >= len(p['atoms']) or n >= len(p['atoms'][j]) or p['atoms'][j][n][0] not in 'NF': return None
            t = fenc_text(e, nv_); k, v = atomval(t)
            p['atoms'][j][n] = (k, v, t)
            continue
        if f['role'] == 'imm' and f.get('zero_by'):
            if keyed(f['zero_by'], new_w) == 'z':
                j, n = (0, 0) if f['slot'] == 'g' else map(int, f['slot'].split('.'))
                if j >= len(p['atoms']) or n >= len(p['atoms'][j]) or p['atoms'][j][n][0] != 'N': return None
                p['atoms'][j][n] = ('N', 0, p['atoms'][j][n][2]); continue
        elif nv_ == ov and not (f['role'] == 'imm' and any(keyed(f.get(k), new_w) != keyed(f.get(k), base_w) for k in ('sign_by', 'neg_by'))):
            continue  # an immediate is printed again when its sign or negation key changes (the Rust runtime's rule)
        r = f['role']
        if r == 'reg':
            j, n = (0, 0) if f['slot'] == 'g' else map(int, f['slot'].split('.'))
            if j >= len(p['atoms']) or n >= len(p['atoms'][j]) or p['atoms'][j][n][0] != f['kind']: return None
            nv_ ^= f.get('xor', 0)
            p['atoms'][j][n] = (f['kind'], nv_, '')
            for sl in f.get('also', []):
                j, n = map(int, sl.split('.'))
                if j < len(p['atoms']) and n < len(p['atoms'][j]) and p['atoms'][j][n][0] == f['kind']: p['atoms'][j][n] = (f['kind'], nv_, '')
        elif r == 'imm':
            j, n = (0, 0) if f['slot'] == 'g' else map(int, f['slot'].split('.'))
            if j >= len(p['atoms']) or n >= len(p['atoms'][j]) or p['atoms'][j][n][0] != 'N': return None
            if not f['weights']: return None
            s = keyed(f.get('sign_by'), new_w)
            v = imm_value(f, new_w, s == 's' if s else f['signed'])
            if keyed(f.get('neg_by'), new_w) == 1: v = -v
            v += f.get('bias', 0)
            p['atoms'][j][n] = ('N', v, p['atoms'][j][n][2])
        elif r == 'mod': pass
        elif r == 'flag':
            for i in range(parts[0][0], parts[0][1] + 1):
                if ((new_w ^ base_w) >> i) & 1:
                    if f['slot'] >= len(p['shapes']): return None
                    s = p['shapes'][f['slot']]; t = model['trans'].get(f'{f["slot"]}|{s}|{i}')
                    if t is None: return None
                    p['shapes'][f['slot']] = t
        elif r == 'none': pass
        else: return None
    for f in fields:
        for j, n, rule in f.get('derive', []):
            d = keyed(rule, new_w)
            if d is None or j >= len(p['atoms']) or n >= len(p['atoms'][j]) or p['atoms'][j][n][0] != f['kind']: continue
            zr = 255 if f['kind'] == 'R' else 7
            fv = pv(new_w, f['parts']) ^ f.get('xor', 0)
            p['atoms'][j][n] = (f['kind'], zr if d == 'Z' or fv == zr else fv + d, '')
    for j, rule in G.get('render', {}).items():
        if not j.isdigit(): continue
        j = int(j); s = keyed(rule, new_w)
        if s is None or j >= len(p['shapes']): continue
        if [ch for ch in s if ch in (PR, PP, PN)] != [{'R': PR, 'P': PP}.get(a[0], PN) for a in p['atoms'][j]]: continue
        p['shapes'][j] = s
    for k, rule in G.get('render', {}).items():
        if not k.startswith('mod:'): continue
        t, has = k[4:], keyed(rule, new_w)
        if has is None: continue
        chains[0] = [x for x in chains[0] if x != t]
        if has: chains.append([t])
    p['mods'] = order_mods(chains, before, {(a, c): n for a, c, n in model.get('before_m', [])})
    # an operand whose value is the one nvdisasm leaves out is not printed
    for j, s, n, c in sorted(model.get('omit', []), reverse=True):
        if len(p['shapes']) == n and p['shapes'][j] == s and len(p['atoms'][j]) == 1 and p['atoms'][j][0][1] == c:
            del p['shapes'][j], p['atoms'][j]
    return p

def composed(f):
    """a field compose() sets at random (and predict() renders from the word)"""
    return f['role'] in ('reg', 'mod', 'flag', 'none') or (f['role'] == 'imm' and f['weights'] and not f['float']) \
        or (f['role'] == 'imm' and f.get('fenc'))

def compose():
    model = json.load(open(R + 'model.json')); nv = all_nv()
    rng = random.Random(20261020)
    btext = {w: t for _, w, _, t in bases_iter()}
    out = []
    for row in sorted(model, key=int):
        M = model[row]; cands = []
        for gname, G in M['groups'].items():
            free = [f for f in G['fields'] if composed(f)]
            for bw in G['bases']:
                w = int(bw, 16); b = at_pc(parse(nv.get(w, btext[w])), w, M.get('pcrel'))
                if b is not None: cands.append((w, b, G, free))
        rng.shuffle(cands)
        made = 0
        for w, b, G, free in cands * PER_ROW:
            if made >= PER_ROW or not free: break
            x = w
            for f in free:
                if f['role'] == 'mod': v = int(rng.choice(list(f['table'])))
                else: v = rng.getrandbits(pwidth(f['parts']))
                x = setpv(x, f['parts'], v)
            if x in nv: continue
            p = predict(M, G, w, b, x)
            if p is None: continue
            out.append((row, x, w, '!REJECT' if keyed(G['render'].get('valid'), x) == 'bad' else render(p, set(M['zero']))))
            made += 1
    with open(R + 'validate.words.tsv', 'w') as o:
        for row, x, w, txt in out: o.write(f'{row}\t{row}\t{x:016x}\t{w:016x}\t{txt}\n')
    print(f'compose: {len(out)} words over {len({o[0] for o in out})} rows', file=sys.stderr)

def canon(t):
    p = parse(t)
    return None if p is None else (p['base'], frozenset(p['mods']), p['shapes'], [[(a[0], a[1]) for a in at] for at in p['atoms']])

def validate():
    nv = all_nv(); per = collections.defaultdict(C)
    with open(R + 'validate-words.tsv', 'w') as o:
        o.write('#row\tword\tbase\texact\tcanonical\tpredicted\tnvdisasm\n')
        for l in open(R + 'validate.words.tsv'):
            row, _, x, w, pred = l.rstrip('\n').split('\t')
            got = nv.get(int(x, 16), '!NOOUT')
            if int(x, 16) in ADDR:  # a PC-relative prediction names the word's own address
                pred = re.sub(r'PC([+-])0x([0-9a-f]+)', lambda m: hex(ADDR[int(x, 16)] + (1 if m.group(1) == '+' else -1) * int(m.group(2), 16)), pred)
            refused = got.startswith('!REJECT') or 'INVALID' in got
            ex = got == pred or (pred == '!REJECT' and refused); ca = ex or (canon(got) is not None and canon(got) == canon(pred))
            per[row]['n'] += 1; per[row]['exact'] += ex; per[row]['canon'] += ca; per[row]['reject'] += got.startswith('!') or 'INVALID' in got
            o.write(f'{row}\t{x}\t{w}\t{int(ex)}\t{int(ca)}\t{pred}\t{got}\n')
    tot = C()
    with open(R + 'validate.tsv', 'w') as o:
        o.write('#row\twords\texact\tcanonical\trejected\texact_rate\tcanonical_rate\n')
        for row in sorted(per, key=int):
            c = per[row]; tot.update(c)
            o.write(f"{row}\t{c['n']}\t{c['exact']}\t{c['canon']}\t{c['reject']}\t{c['exact'] / c['n']:.2f}\t{c['canon'] / c['n']:.2f}\n")
    full = sum(1 for c in per.values() if c['exact'] == c['n'])
    print(f"validate: {len(per)} rows, {tot['n']} words, exact {tot['exact']} ({tot['exact'] / max(1, tot['n']):.1%}), "
          f"canonical {tot['canon']}, rejected {tot['reject']}, rows all-exact {full}", file=sys.stderr)

# ---------------------------------------------------------------- export
# One stable table of everything the layouts know, for a decoder generator
# outside this repo. A variant is one base word of a row's operand form: its
# fixed bits (opcode, form, enumerated operands, immediates with no value
# model) are the mask and value; every other field is free, and predict()
# renders any word that keeps the fixed bits from the base's parsed text.
# The table holds no paths, hosts or program names: rows, bits, value tables
# and texts only. Each learned rendering rule is a typed record: an
# immediate's per-bit weights, constant, sign, negation and zeroing; a float's
# encoding; a register printed twice, inverted, or as another register atom's
# pair; a modifier no field owns; the words the disassembler refuses, with
# their backoff; operands left out at one value; modifier order from rows
# sharing a mnemonic. Their key bits are free, as the rules decide them.
EXPORT_VERSION = 'decoder-formats/3'
FREE = ('reg', 'mod', 'flag', 'none')

def free_field(f):
    if f['role'] in FREE: return True
    if f['role'] != 'imm': return False
    return bool(f.get('fenc')) if f['float'] else bool(f.get('weights'))

def pmask(parts):
    m = 0
    for lo, hi in parts: m |= ((1 << (hi - lo + 1)) - 1) << lo
    return m

def ebits(parts):
    """parts in value order, least significant first"""
    return ','.join(f'{hi}:{lo}' for lo, hi in parts)

def eshape(s):
    for ch, k in ((PR, 'R'), (PP, 'P'), (PN, 'N')): s = s.replace(ch, '\\' + k)
    return s

def eoperand(shape, atoms):
    """shape with each placeholder carrying its atom: \\R5; \\P7; \\Nx-16; \\Nd3; \\F1.5;"""
    it, o = iter(atoms), ''
    for ch in shape:
        if ch in (PR, PP, PN):
            k, v, raw = next(it)
            if k == 'N': o += f"\\N{'x' if raw.lstrip('-').startswith('0x') else 'd'}{v};"
            elif k == 'F': o += f'\\F{raw};'
            else: o += f'\\{k}{v};'
        else: o += ch
    return o

def ekey(rule):
    """key components joined by |: bits, or nz:bits when only zero or not counts; - for none"""
    nz = set(rule.get('nz', ()))
    return '|'.join(('nz:' if i in nz else '') + ebits(c) for i, c in enumerate(rule['key'])) or '-'

def ekeyed(rule, fmt):
    """key, value=result entries, and *=result for key values never seen"""
    out = [ekey(rule)] + [f'{k}={fmt(v)}' for k, v in sorted(rule['table'].items())]
    if rule.get('default') is not None: out.append(f"*={fmt(rule['default'])}")
    return out

def escale(f):
    """<<s for a regular immediate, else w: each bit's weight exponent in value order"""
    if f['shift'] is not None and f['shift'] >= 0: return f"<<{f['shift']}"
    return 'w:' + ','.join(str(x.bit_length() - 1) for x in f['weights'])

SIGN = {'s': 'twos', 'u': 'plain'}
ZERO = {'z': 'zero', 'v': 'value'}

def refused_by(G, w):
    """True when the form's learned refusals say the disassembler rejects w"""
    return keyed(G.get('render', {}).get('valid'), w) == 'bad'

def export():
    """export R OUT ORACLE...: OUT/decoder-formats.tsv and OUT/decoder-format-samples.tsv.
    ORACLE files (word<TAB>text, e.g. an opcode-space or row-sweep nv.tsv) join
    the run's own answers as held samples. Each variant's measured rate is
    predict() against every held word it renders (first variant in table order
    whose fixed bits match, whose prediction exists and which the form's
    refusals leave)."""
    import hashlib
    out = sys.argv[3].rstrip('/') + '/'
    model = json.load(open(R + 'model.json'))
    btext = {w: t for _, w, _, t in bases_iter()}
    # every distinct text a word was given: a word printed two ways (a
    # PC-relative target names the address it was disassembled at) is a
    # word no variant can print exactly
    pool = collections.defaultdict(list)
    def held(w, t):
        if t not in pool[w]: pool[w].append(t)
    for w, t in json.load(open(R + 'known.json')).items(): held(int(w), t)
    for path in [R + n for n in ('samples.nv.tsv', 'flips.nv.tsv', 'pairs.nv.tsv', 'tables.nv.tsv', 'validate.nv.tsv')] + sys.argv[4:]:
        if not os.path.exists(path): continue
        for l in open(path):
            w, _, t = l.rstrip('\n').partition('\t')
            if w and not w.startswith('#'): held(int(w, 16), t.partition('\t@')[0])
    variants = []  # (row, gname, G, M, base_w, parsed base, mask, value)
    for row in sorted(model, key=int):
        M = model[row]
        for gname, G in M['groups'].items():
            mask = 0
            for f in G['fields']:
                if not free_field(f): mask |= pmask(f['parts'])
            seen = set()
            for bw in G['bases']:
                w = int(bw, 16)
                if w & mask in seen: continue
                b = parse(btext[w])
                if b is None: continue
                seen.add(w & mask); variants.append((row, gname, G, M, w, b, mask, w & mask))
    by_prefix = collections.defaultdict(list)
    for i, v in enumerate(variants):
        mm, vv = v[6] >> 48, v[7] >> 48
        free = [k for k in range(16) if not (mm >> k) & 1]
        for bits in range(1 << len(free)):
            p = vv
            for j, k in enumerate(free):
                if (bits >> j) & 1: p |= 1 << k
            by_prefix[p].append(i)
    n, ex = C(), C()
    zero = {row: set(M['zero']) for row, M in model.items()}
    with open(out + 'decoder-format-samples.tsv', 'w') as o:
        o.write(f'# {EXPORT_VERSION} samples: every held word inside some variant\'s fixed bits, with each disassembler text it was given\n')
        for x in sorted(pool):
            ts = pool[x]; matched = False
            for i in by_prefix.get(x >> 48, ()):
                row, gname, G, M, w, b, mask, value = variants[i]
                if x & mask != value: continue
                matched = True
                if refused_by(G, x): continue
                p = predict(M, G, w, b, x)
                if p is None: continue
                n[i] += 1; ex[i] += ts == [render(p, zero[row])]
                break
            if matched:
                for t in ts: o.write(f'{x:016x}\t{t}\n')
    digest = hashlib.sha256(open(R + 'model.json', 'rb').read()).hexdigest()
    L = []
    for row in sorted(model, key=int):
        M = model[row]
        z = sorted(M['zero'])
        L.append(('zero', row, z[0] if z else 'RZ'))
        for a, c, k in sorted(M['before']): L.append(('order', row, a, c, k))
        for a, c, k in sorted(M.get('before_m', [])): L.append(('order-other', row, a, c, k))
        for j, s, n_, c in M.get('omit', []): L.append(('omit', row, j, eshape(s), n_, c))
        for key, s in sorted(M['trans'].items()):
            slot, rest = key.split('|', 1); shape, bit = rest.rsplit('|', 1)
            L.append(('flag', row, slot, eshape(shape), bit, eshape(s)))
        for gname, G in M['groups'].items():
            for k, f in enumerate(G['fields']):
                r = f['role']; bits = ebits(f['parts'])
                if r == 'reg':
                    L.append(('field', row, gname, k, 'reg', bits, f['kind'], f['slot'], f.get('xor', 0), *f.get('also', [])))
                    for j, n_, rule in f.get('derive', []): L.append(('derive', row, gname, k, f'{j}.{n_}', *ekeyed(rule, str)))
                elif r == 'imm' and free_field(f) and f['float']:
                    L.append(('field', row, gname, k, 'float', bits, f['slot'], f['fenc']))
                    if f.get('enc_by'): L.append(('rule', row, gname, k, 'enc', *ekeyed(f['enc_by'], str)))
                elif r == 'imm' and free_field(f):
                    L.append(('field', row, gname, k, 'imm', bits, f['slot'], escale(f), f.get('bias', 0), 'twos' if f['signed'] else 'plain'))
                    if f.get('sign_by'): L.append(('rule', row, gname, k, 'sign', *ekeyed(f['sign_by'], SIGN.get)))
                    if f.get('neg_by'): L.append(('rule', row, gname, k, 'neg', *ekeyed(f['neg_by'], str)))
                    if f.get('zero_by'): L.append(('rule', row, gname, k, 'zero', *ekeyed(f['zero_by'], ZERO.get)))
                elif r == 'mod':
                    L.append(('field', row, gname, k, 'mod', bits, '.'.join(f['tokens']) or '-',
                              *[f"{v}={'.'.join(f['table'][v]) or '-'}" for v in sorted(f['table'], key=int)]))
                elif r == 'flag': L.append(('field', row, gname, k, 'flag', bits, f['slot']))
                elif r == 'none': L.append(('field', row, gname, k, 'none', bits))
                elif r == 'enum':
                    L.append(('field', row, gname, k, 'fixed-enum', bits, f['slot'],
                              *[f'{v}={eshape(f["table"][v])}' for v in sorted(f.get('table', {}), key=int)]))
                else: L.append(('field', row, gname, k, 'fixed-' + r, bits))
            rs = G.get('render', {})
            for j, rule in sorted(((j, r) for j, r in rs.items() if j.isdigit()), key=lambda t: int(t[0])):
                L.append(('render', row, gname, j, *ekeyed(rule, eshape)))
            for t, rule in sorted((k[4:], r) for k, r in rs.items() if k.startswith('mod:')):
                L.append(('token', row, gname, t, *ekeyed(rule, str)))
            if rs.get('valid'):
                v = rs['valid']; L.append(('refuse', row, gname, *ekeyed(v, str)))
                if 'pure' in v:
                    for i, pm in enumerate(v['pure']):
                        if pm: L.append(('refuse-alone', row, gname, i, *[f'{x}={a}' for x, a in sorted(pm.items(), key=lambda t: int(t[0]))]))
                    L.append(('refuse-rest', row, gname, v.get('rest') or v.get('default') or 'ok'))
    for i, (row, gname, G, M, w, b, mask, value) in enumerate(variants):
        ops = [eoperand(s, a) for s, a in zip(b['shapes'], b['atoms'])]
        L.append(('variant', i, row, gname, f'{mask:016x}', f'{value:016x}', '%016x' % w, n[i], ex[i],
                  'glued' if b['glued'] else 'spaced', b['base'], '.'.join(b['mods']) or '-', *ops))
    tot, tex = sum(n.values()), sum(ex.values())
    with open(out + 'decoder-formats.tsv', 'w') as o:
        o.write(f'# {EXPORT_VERSION}\n# Maxwell (SM52) instruction formats inferred from disassembler answers to\n'
                f'# bit flips of known words. model sha256 {digest}\n'
                f'# {len(variants)} variants over {len(model)} rows; {tot} held words rendered, {tex} exactly.\n'
                '# Bits: parts hi:lo in value order, least significant part first, comma-joined.\n'
                '# Texts: \\R \\P \\N mark register, predicate and number atoms in an operand shape;\n'
                '#   in a variant template each carries its value: \\R5; \\P7; \\Nx-16; (hex) \\Nd3; (decimal) \\F1.5; (float).\n'
                '# Keys: selector and register bits joined by | (nz:bits counts only zero 0 or not 1; - no\n'
                '#   key); entries value=result with the key values comma-joined; *=result for key values\n'
                '#   never seen (none: the rule says nothing there).\n'
                '# zero    row text                        how [RZ+0x0] prints in this row\n'
                '# order   row a b count                   modifier a printed before b this often\n'
                '# order-other row a b count               the same over rows printing the mnemonic, for pairs\n'
                '#                                         the row never printed together\n'
                '# omit    row operand shape count value   in a text of count operands, operand (of shape) is\n'
                '#                                         left out when its one atom is value (RZ 255, PT 7)\n'
                '# flag    row slot shape bit shape        flipping bit turns operand slot\'s shape into the next\n'
                '# field   row form index role bits ...\n'
                '#   reg   file slot xor [slot...]         printed xor xor at the first slot, and again at each other\n'
                '#   imm   slot scale bias sign            value = sum of set bits\' weights (scale <<s: bit k weighs\n'
                '#                                         2^(k+s); w:e,e,..: bit k weighs 2^e_k), the top bit negative\n'
                '#                                         when sign is twos, negated when its neg rule says 1, plus bias\n'
                '#   float slot encoding                   [-]W<<s: the field is bits s up of a W-bit IEEE float (s < 0:\n'
                '#                                         the field\'s bits from -s up); - flips the sign bit\n'
                '#   mod   tokens value=names...;  flag slot;  none\n'
                '#   fixed-* ...                           part of the variant mask (fixed-enum: slot value=shape...)\n'
                '# rule    row form index kind key entries  a rule of field index: sign (twos|plain; else the field\'s),\n'
                '#                                         neg (1 negates; else 0), enc (encoding; else the field\'s),\n'
                '#                                         zero (zero prints 0; else value)\n'
                '# derive  row form index slot key entries  register atom slot prints field index\'s value plus\n'
                '#                                         the entry, or the zero register (Z), or itself (no entry)\n'
                '# render  row form slot key value=shape... operand shape decided by selector values\n'
                '# token   row form name key value=0|1...  a modifier no field owns, printed where the key says 1\n'
                '# refuse  row form key value=ok|bad...    words the disassembler rejects (bad): no variant prints them\n'
                '# refuse-alone row form k value=ok|bad... key values never seen: component k\'s values that decide\n'
                '#                                         alone; two disagreeing give the refuse fallback\n'
                '# refuse-rest row form ok|bad             ... and when no component decides\n'
                '# variant id row form mask value base samples exact glued|spaced mnemonic mods guard operands...\n')
        for t in L: o.write('\t'.join(map(str, t)) + '\n')
    full = sum(1 for i in range(len(variants)) if n[i] and ex[i] == n[i])
    print(f'export: {len(variants)} variants, {tot} held words rendered, {tex} exact ({tex / max(1, tot):.1%}), '
          f'{full} variants exact on every word', file=sys.stderr)

{'pick': pick, 'pairs': pairs, 'pc1': pc1, 'pc2': pc2, 'layout': layout, 'compose': compose, 'validate': validate, 'export': export}[sys.argv[1]]()
