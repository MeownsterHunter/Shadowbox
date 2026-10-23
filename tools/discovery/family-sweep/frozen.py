"""The frozen evaluation set: one fixed held-out word list per row (family).

THE IDEA
  infer.py's validate composes its test words from the run's own layout, so
  every layout change redraws them and two runs' exact-match numbers are
  measured on different words. The frozen set is drawn once, with a recorded
  seed, and every run is measured on it too: same words, same nvdisasm
  answers (the answer store holds them), so the numbers compare. The per-run
  composed set stays as the second number: it follows the current layout
  into fields the frozen set never varied, which is what catches a model
  fitted to the frozen words.

  make  Draws the set with infer.py's own compose() on a finished run's
        layout, seed SEED instead of compose's, PER words per row: every
        register, immediate, modifier, flag and no-effect field set at random
        on one of the row's bases, never a word that run knew. Writes
        <store>/frozen-eval/words.tsv (row, row, word, base) and meta.txt
        (seed, words per row, source run, model digest, date). Refuses to
        overwrite: a new set is a new name, and the old numbers stop
        comparing.
  eval  For run dir R: R/frozen/ links R's answer files, and gets
        validate.words.tsv = every frozen word R does not already know
        (skipped and counted otherwise) with R's prediction for it, made the
        way compose() makes one: base parse, predict() on a form group,
        render(). The model is the one of R that covers the word, not the
        base the word was drawn on: a run picks one base per operand shape
        and a parser change merges or splits shapes, so the drawn base may
        be gone while its form is still modelled. A base of the word's row
        covers it when the word keeps every bit of the base outside the
        fields compose() sets at random (composed()), so predict() renders
        the word from that base and its group. The group the word's own
        form bits (the row's fmask) name comes first, since another group
        can read those bits as a field and cover it too; then the nearest
        base (fewest differing bits, then lowest word). The first whose
        prediction exists wins; the column 'base' of the scored words
        names it. A word no model of R covers, or that no covering model
        can predict, is predicted as '!UNPREDICTED' and counts as a miss.
        family-infer.sh then asks nvdisasm (the store) for frozen/validate.words.tsv and
        runs infer.py validate on R/frozen, so the scoring is validate's
        own, and a run's score depends only on what its models print.
usage: frozen.py make RUN [SEED] [PER]    frozen.py eval RUN
  The set lives in <store>/frozen-eval (<store> as in store.py); FROZEN_SET
  names another dir (family-infer.sh uses it for a local copy of a store
  kept on another host). A set made elsewhere is copied there by hand.
"""
import sys, os, ast, json, hashlib, random, time, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get('DISCOVERY_STORE') or os.path.join(os.environ.get('DISCOVERY_WORK', 'target/discovery'), 'store')
SET = os.environ.get('FROZEN_SET') or STORE + '/frozen-eval'

def infer(R):
    """infer.py's definitions (its command dispatch, the last statement, left out) with run dir R"""
    tree = ast.parse(open(HERE + '/infer.py').read())
    tree.body = tree.body[:-1]
    sys.argv = ['infer.py', '-', R]
    ns = {'__name__': 'infer', '__file__': HERE + '/infer.py'}
    exec(compile(tree, HERE + '/infer.py', 'exec'), ns)
    return ns

def link_run(run, d, skip=('validate',)):
    """d links every answer file of run: a link left pointing elsewhere (a run
    dir copied from another run keeps the old run's links) points here again"""
    os.makedirs(d, exist_ok=True)
    for n in os.listdir(run):
        p = os.path.abspath(os.path.join(run, n)); q = os.path.join(d, n)
        if not os.path.isfile(p) or n.startswith(skip): continue
        if os.path.islink(q) and os.readlink(q) != p: os.unlink(q)
        if not os.path.lexists(q): os.symlink(p, q)

def make(run, seed=20261021, per=32):
    if os.path.exists(SET + '/words.tsv'): sys.exit(f'{SET}/words.tsv exists: the frozen set is drawn once')
    os.makedirs(SET, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(run)), prefix='frozen-make.')
    link_run(run, tmp)
    ns = infer(tmp + '/')
    class Shim: Random = staticmethod(lambda _s: random.Random(seed))
    ns['random'] = Shim; ns['PER_ROW'] = per
    ns['compose']()
    rows = []
    for l in open(tmp + '/validate.words.tsv'):
        row, _, x, w, _ = l.rstrip('\n').split('\t'); rows.append((row, x, w))
    with open(SET + '/words.tsv', 'w') as o:
        o.write('#row\trow\tword\tbase\n')
        for row, x, w in rows: o.write(f'{row}\t{row}\t{x}\t{w}\n')
    digest = hashlib.sha256(open(run + '/model.json', 'rb').read()).hexdigest()
    with open(SET + '/meta.txt', 'w') as o:
        o.write(f'seed\t{seed}\nper_row\t{per}\nsource_run\t{os.path.basename(os.path.abspath(run))}\nmodel_sha256\t{digest}\n'
                f"made\t{time.strftime('%Y-%m-%d %H:%M:%S')}\nwords\t{len(rows)}\nrows\t{len({r[0] for r in rows})}\n")
    shutil.rmtree(tmp)
    print(f'frozen set: {len(rows)} words over {len({r[0] for r in rows})} rows (seed {seed}, {per} per row) -> {SET}', file=sys.stderr)

def covering(ns, M, x):
    """(group, base) of row model M whose form holds x, best first: x keeps every bit of the
    base outside the fields compose() sets. The group x's own form bits name comes first
    (another group can hold x too when it reads those bits as a field), then the base
    with the fewest bits differing from x, then the lowest word."""
    out, fm = [], M['fmask']
    own = f'{x & fm:x}' if fm else '-'
    for gname, G in M['groups'].items():
        free = 0
        for f in G['fields']:
            if ns['composed'](f): free |= ns['pmask'](f['parts'])
        for bw in G['bases']:
            w = int(bw, 16)
            if (x ^ w) & ~free == 0: out.append((gname != own, bin(x ^ w).count('1'), w, G))
    return [(G, w) for *_, w, G in sorted(out, key=lambda c: c[:3])]

def predict_text(ns, nv, btext, M, G, w, x):
    """the text compose() would predict for x composed on base w of group G, or None"""
    b = ns['parse'](btext[w])
    if 'at_pc' in ns: b = ns['at_pc'](ns['parse'](nv.get(w, btext[w])), w, M.get('pcrel'))
    if b is None: return None
    try: p = ns['predict'](M, G, w, b, x)
    except (TypeError, KeyError, ValueError, IndexError):
        return None  # predict() assumes compose's own field choice (e.g. an immediate with no known scale)
    if p is None: return None
    valid = G.get('render', {}).get('valid')
    if valid and ns['keyed'](valid, x) == 'bad': return '!REJECT'
    return ns['render'](p, set(M['zero']))

def evaluate(run):
    d = run.rstrip('/') + '/frozen'
    for n in ('validate.words.tsv', 'validate.nv.tsv'):
        if os.path.lexists(f'{d}/{n}'): os.unlink(f'{d}/{n}')
    link_run(run, d)
    ns = infer(d + '/')
    model = json.load(open(d + '/model.json')); nv = ns['all_nv']()
    btext = {w: t for _, w, _, t in ns['bases_iter']()}
    n = known = uncovered = unpred = moved = 0
    with open(d + '/validate.words.tsv', 'w') as o:
        for l in open(SET + '/words.tsv'):
            if l.startswith('#'): continue
            row, _, xs, ws = l.rstrip('\n').split('\t'); x = int(xs, 16)
            if x in nv: known += 1; continue
            cover = covering(ns, model[row], x) if row in model else []
            t = None
            for G, w in cover:
                t = predict_text(ns, nv, btext, model[row], G, w, x)
                if t is not None: moved += f'{w:016x}' != ws; ws = f'{w:016x}'; break
            if t is None: t = '!UNPREDICTED'; uncovered += not cover; unpred += bool(cover)
            o.write(f'{row}\t{row}\t{xs}\t{ws}\t{t}\n'); n += 1
    print(f'frozen set: {n} words to score, {known} skipped (known to this run), {moved} on another base than drawn, '
          f'{uncovered} covered by no model and {unpred} unpredictable (both scored as misses)', file=sys.stderr)

if __name__ == '__main__':
    if sys.argv[1] == 'make': make(sys.argv[2], *[int(a) for a in sys.argv[3:5]])
    elif sys.argv[1] == 'eval': evaluate(sys.argv[2])
    else: sys.exit('usage: frozen.py make RUN [SEED] [PER] | eval RUN')
