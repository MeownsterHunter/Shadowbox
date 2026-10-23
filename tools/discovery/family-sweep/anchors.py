"""Pick a per-row operand anchor that shadowbox can decode.

The manifest baseline puts 0 in every operand field. Some decoders only
admit a family with RZ/PT operands or a real attribute address (IPA needs
RZ sources, PT predicate and attribute >= 0x80), so a sweep over the
baseline alone reports whole rows as missing. For each row this tries
every combination of {baseline, RZ} per register field, {baseline, PT}
per predicate field and {baseline, 0x80} per attribute-address field
(at most 7 fields), and keeps the one where shadowbox decodes the most
of 129 modifier samples. Overrides below hold nvdisasm-valid choices.
usage: anchors.py ROWS_JSON SBDECODE OUT_JSON
"""
import json, random, re, struct, subprocess, itertools, sys
d = json.load(open(sys.argv[1])); SB = sys.argv[2]
# ALD: nvdisasm rejects a[Rx+imm] unless .P; a[imm] needs Ra = RZ.
OVERRIDE = {'1885': 0xefd800000807ff00}
def bits(m): return [i for i in range(64) if m >> i & 1]
res = {}
for k, v in sorted(d.items(), key=lambda kv: int(kv[0])):
    opf = []
    for n, m in v['fields'].items():
        last = n.split('/')[-1]
        if n.startswith('T(pred)'): continue
        m &= ~v['mask']
        if not m: continue
        c = bin(m).count('1')
        if re.match(r'reg\d\d_bf', last) and c == 8: opf.append((n, m, 0xff))
        elif re.match(r'pred\d\d_bf', last) and c == 3: opf.append((n, m, 7))
        elif last.startswith('amem') and c == 10: opf.append((n, m, 0x80))
    opf = opf[:7]
    rng = random.Random(int(k))
    named = v['mod']; nb = bits(named)
    mods = [sum(1 << b for b in nb if rng.random() < 0.5) for _ in range(128)] + [0]
    cands = []
    for choice in itertools.product([0, 1], repeat=len(opf)):
        a = v['base']
        for (n, m, val), c in zip(opf, choice):
            if c: a = (a & ~m) | ((val << bits(m)[0]) & m)
        cands.append((choice, a))
    words = [(a & ~named) | x for _, a in cands for x in mods]
    out = subprocess.run([SB], input=b''.join(struct.pack('<Q', w) for w in words),
                         capture_output=True, check=True).stdout.decode().splitlines()
    n = len(mods); best = None
    for i, (choice, a) in enumerate(cands):
        score = sum(1 for l in out[i*n:(i+1)*n] if not l.endswith('Unknown'))
        key = (score, -sum(choice))
        if best is None or key > best[0]: best = (key, a)
    res[k] = dict(anchor=OVERRIDE.get(k, best[1]), score=best[0][0])
json.dump(res, open(sys.argv[3], 'w'))
print(f'anchors: {sum(1 for k in res if res[k]["anchor"] != d[k]["base"])} rows moved off baseline', file=sys.stderr)
