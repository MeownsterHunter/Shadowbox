"""Split a family-infer run into row units for the layout passes, and merge them.

THE IDEA
  infer.py layout works row by row: every dict it builds is keyed by row,
  and a row's fields, tables and conflicts come only from that row's bases,
  their single and pair flips, and the enumeration words asked for that
  row. So the layout passes (layout, ask the new table words, layout again
  until nothing is new) can run on any group of rows, on any machine, and
  the run's result is the rows' results side by side. A unit is such a
  group; family-infer.sh hands units to workers on every host as they free
  up (work stealing), then merges them here.

  split RUN N  RUN/units/uNNN/: bases.tsv (that unit's rows), flips.nv.tsv
               (the run's answer, as infer.py's all_nv() sees it, for every
               base word, single flip and pair flip of those bases, with
               the @address suffix where it has one), pc.rel.tsv (its rows'
               PC-relative atoms) and an empty known.json, so a unit carries
               only what its rows read.
               Rows are packed into about N units by base count (a row with
               more bases than RUN's share stays alone); RUN/units/todo/ gets
               one file per unit, named so the heaviest is taken first.
  merge RUN    layouts.tsv, conflicts.tsv and model.json in row order, as one
               layout pass over every row writes them; tables.words.tsv and
               tables.nv.tsv (appended to RUN's own, a cache) concatenated;
               tables.new/ask.words.tsv empty, as after the last pass.
"""
import sys, os, json, collections

def load_nv(R, names, nv):
    for n in names:
        if os.path.exists(R + n):
            for l in open(R + n):
                w, _, t = l.rstrip('\n').partition('\t')
                if w: nv[int(w, 16)] = t
    return nv

def split(R, n):
    nv = {int(k): v for k, v in json.load(open(R + 'known.json')).items()}
    load_nv(R, ('samples.nv.tsv', 'flips.nv.tsv', 'pairs.nv.tsv', 'tables.nv.tsv', 'validate.nv.tsv',
                'pc1.nv.tsv', 'pc2.nv.tsv'), nv)  # infer.py all_nv()
    rel = collections.defaultdict(list)
    if os.path.exists(R + 'pc.rel.tsv'):
        for l in open(R + 'pc.rel.tsv'): rel[l.split('\t', 1)[0]].append(l)
    rows = collections.defaultdict(list)
    for l in open(R + 'bases.tsv'): rows[l.split('\t', 1)[0]].append(l)
    total = sum(len(v) for v in rows.values()); share = max(1, total // max(1, n))
    units, cur, w = [], [], 0
    for row in sorted(rows, key=lambda r: (-len(rows[r]), int(r))):
        cur.append(row); w += len(rows[row])
        if w >= share: units.append(cur); cur, w = [], 0
    if cur: units.append(cur)
    os.makedirs(R + 'units/todo', exist_ok=True)
    for i, us in enumerate(units):
        d = R + f'units/u{i:03d}/'; os.makedirs(d, exist_ok=True)
        need = set()
        with open(d + 'bases.tsv', 'w') as o:
            for row in sorted(us, key=int):
                for l in rows[row]:
                    o.write(l); b = int(l.split('\t')[1], 16); need.add(b)
                    for j in range(64): need.add(b ^ (1 << j))
                    for j in range(63): need.add(b ^ (3 << j))
        with open(d + 'flips.nv.tsv', 'w') as o:
            for x in sorted(need):
                if x in nv: o.write(f'{x:016x}\t{nv[x]}\n')
        with open(d + 'pc.rel.tsv', 'w') as o:
            for row in sorted(us, key=int): o.writelines(rel[row])
        json.dump({}, open(d + 'known.json', 'w'))
        with open(d + 'rows', 'w') as o: o.write(','.join(sorted(us, key=int)) + '\n')
        nb = sum(len(rows[r]) for r in us)
        open(R + f'units/todo/{10**7 - nb:07d}-u{i:03d}', 'w').close()
    print(f'units: {len(rows)} rows, {total} bases in {len(units)} units (about {share} bases each)', file=sys.stderr)

def blocks(path):
    """row -> its lines, in file order (a file lists each row's lines together)"""
    out = collections.defaultdict(list)
    for l in open(path):
        if not l.startswith('#'): out[l.split('\t', 1)[0]].append(l)
    return out

def merge(R):
    us = sorted(d for d in os.listdir(R + 'units') if d.startswith('u') and os.path.isdir(R + 'units/' + d))
    lay, con, model = {}, {}, {}
    for u in us:
        d = R + 'units/' + u + '/'
        for row, ls in blocks(d + 'layouts.tsv').items(): lay[row] = ls
        for row, ls in blocks(d + 'conflicts.tsv').items(): con[row] = ls
        model.update(json.load(open(d + 'model.json')))
    with open(R + 'layouts.tsv', 'w') as o:
        o.write('#row\tform\tfield\tbits\trole\tnames\n')
        for row in sorted(lay, key=int): o.writelines(lay[row])
    with open(R + 'conflicts.tsv', 'w') as o:
        o.write('#row\tform\tbit\tkind\tcounts\texample_base\n')
        for row in sorted(con, key=int): o.writelines(con[row])
    json.dump({row: model[row] for row in sorted(model, key=int)}, open(R + 'model.json', 'w'))
    with open(R + 'tables.words.tsv', 'w') as o:
        for u in us:
            p = R + 'units/' + u + '/tables.words.tsv'
            if os.path.exists(p): o.writelines(open(p))
    with open(R + 'tables.nv.tsv', 'a') as o:
        for u in us:
            p = R + 'units/' + u + '/tables.nv.tsv'
            if os.path.exists(p): o.writelines(open(p))
    for n in ('tables.new.words.tsv', 'tables.ask.words.tsv'): open(R + n, 'w').close()
    nf = sum(len(v) for v in lay.values()); nc = sum(len(v) for v in con.values())
    ng = sum(len(m['groups']) for m in model.values())
    print(f'merge: {len(us)} units, {len(model)} rows, {ng} form groups, {nf} fields, {nc} conflicts', file=sys.stderr)

if __name__ == '__main__':
    R = sys.argv[2].rstrip('/') + '/'
    if sys.argv[1] == 'split': split(R, int(sys.argv[3]))
    elif sys.argv[1] == 'merge': merge(R)
    else: sys.exit('usage: units.py split RUN N | merge RUN')
