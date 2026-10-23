"""The shared nvdisasm answer store: every word nvdisasm was ever asked, once.

THE IDEA
  nvdisasm's text for a word is the ground truth the discovery tools build
  on, and it is a pure function of (the nvdisasm binary, how it is invoked,
  where the word sits, the word). Without a store every run asks again.
  Here every answer goes into one insert-only store; an ask takes what the
  store already holds and sends only the missing words to nvdisasm. The
  per-run .nv.tsv is written exactly as before (same lines, same format), so
  nothing downstream changes, and the store never changes which words a run
  generates: it only answers them.

KEY AND VALUE (sqlite, <store>/answers.sqlite; <store> is $DISCOVERY_STORE,
  default $DISCOVERY_WORK/store)
  key.version  'nvdisasm sha256:<digest of the binary>' of the nvdisasm that
               answers (ND names it; NVSTORE_VERSION passes a digest computed
               elsewhere, when the store lives on another host). hosts.sh
               refuses to ask a host whose nvdisasm has another digest.
  key.flags    FLAGS below: the command line, the 32-byte group the word sits
               in, and how nv.py turns a refusal into text. Anything else
               that could change the text must go in here.
  answer.word  the 64-bit word (stored as signed int64).
  answer.addr  the address nvdisasm printed the word at, or -1 when unknown
               (answers seeded from files made without NVADDR=1). A
               PC-relative target is the word's address plus a value, so the
               text is only an answer *at that address*. An ask that wants
               addresses (NVADDR=1) only takes answers with one; an ask that
               does not (the plain nv.py contract: an arbitrary address)
               takes any.
  answer.text  nvdisasm's text, '!REJECT <message>' or '!NOOUT', exactly as
               nv.py writes it (without the @address suffix).
  answer.source  the run file or step that produced it.
  Rows are only ever inserted (triggers refuse UPDATE and DELETE). One word
  can hold several answers (different addresses; seeded texts made at
  unknown addresses that differ); a lookup takes the oldest that fits.
  Inserting the same answer twice is a no-op (UNIQUE key, word, addr, text),
  so several hosts may put the same words at once.

SPOT RECHECK
  Every take sends a random sample of its hits (1%, at least 200) to the live
  nvdisasm (ND, on the machine that runs the take) in nv.py batches and
  compares. A text that differs at a different address is asked again alone
  at the stored address (or, when that is unknown, at a second address: if
  the two live texts differ the word is address-dependent and cannot be
  compared). Any mismatch fails the run loudly with the key: a mismatch
  means the key is missing something that changes the text.

USAGE (callers: hosts.sh nv_store_take / nv_store_put)
  ND=<nvdisasm> store.py take WORDS_TSV OUT_NV
      lookup, then recheck. WORDS_TSV column 3 is the word. Writes OUT_NV.hit
      (the store's answers, nv.tsv lines; '<TAB>@addr' appended when
      NVADDR=1) and WORDS_TSV.ask (the lines whose word is missing, one per
      word). Prints the hit rate, appends it to <store>/take.log, rechecks a
      sample. NVSTORE_LIVE=1 answers nothing from the store (a step that
      re-asks known words on purpose); put still stores what nvdisasm says.
  ND=... store.py put WORDS_TSV OUT_NV SOURCE
      insert, then assemble. OUT_NV holds nv.py's answers to WORDS_TSV.ask,
      asked with NVADDR=1. Appends them to the store, then rewrites OUT_NV as
      hits + new answers in the caller's format (no address unless NVADDR=1).
  The halves, for a store on another host (only the store host opens the
  sqlite; the asking machine rechecks with its own nvdisasm):
    store.py lookup WORDS_TSV OUT_NV   (store host) .hit, .ask, and OUT_NV.check
                                       (the recheck sample, with addresses)
    store.py recheck OUT_NV LABEL      (asking machine) asks OUT_NV.check live
    store.py insert OUT_NV SOURCE      (store host) stores OUT_NV's answers
    store.py assemble WORDS_TSV OUT_NV (asking machine) OUT_NV = hits + new
  ND=... store.py seed NV_TSV [accepted]
      Appends an existing run file's answers (address from its '@' suffix,
      else -1). 'accepted' skips its refusals (files whose refusal text used
      an older convention). Only for files made with this version and FLAGS.
"""
import sys, os, sqlite3, hashlib, random, math, importlib.util, time, tempfile
STORE = os.environ.get('DISCOVERY_STORE') or os.path.join(os.environ.get('DISCOVERY_WORK', 'target/discovery'), 'store')
FLAGS = ('-b SM52 -raw; group [0x001fc400fe2007f6, WORD, 0x50b0000000070f00, 0x50b0000000070f00], '
         'WORD at 32*i+8; refusal = first stderr line, address removed, 200 chars')
WANT_ADDR = os.environ.get('NVADDR') == '1'

def s64(w): return w - (1 << 64) if w >= 1 << 63 else w
def u64(w): return w + (1 << 64) if w < 0 else w

def version():
    if os.environ.get('NVSTORE_VERSION'): return 'nvdisasm sha256:' + os.environ['NVSTORE_VERSION']
    h = hashlib.sha256()
    with open(os.environ['ND'], 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return 'nvdisasm sha256:' + h.hexdigest()

def db():
    os.makedirs(STORE, exist_ok=True)
    c = sqlite3.connect(STORE + '/answers.sqlite', timeout=600)
    c.execute('PRAGMA journal_mode=WAL')
    c.executescript('''
      CREATE TABLE IF NOT EXISTS key(id INTEGER PRIMARY KEY, version TEXT NOT NULL, flags TEXT NOT NULL, UNIQUE(version, flags));
      CREATE TABLE IF NOT EXISTS source(id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, added TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS answer(key INTEGER NOT NULL, word INTEGER NOT NULL, addr INTEGER NOT NULL,
                                        text TEXT NOT NULL, source INTEGER NOT NULL, UNIQUE(key, word, addr, text));
      CREATE TRIGGER IF NOT EXISTS answer_no_update BEFORE UPDATE ON answer BEGIN SELECT RAISE(ABORT, 'append-only'); END;
      CREATE TRIGGER IF NOT EXISTS answer_no_delete BEFORE DELETE ON answer BEGIN SELECT RAISE(ABORT, 'append-only'); END;
      CREATE TRIGGER IF NOT EXISTS key_no_update BEFORE UPDATE ON key BEGIN SELECT RAISE(ABORT, 'append-only'); END;
      CREATE TRIGGER IF NOT EXISTS key_no_delete BEFORE DELETE ON key BEGIN SELECT RAISE(ABORT, 'append-only'); END;''')
    return c

def key_id(c, ver):
    c.execute('INSERT OR IGNORE INTO key(version, flags) VALUES (?, ?)', (ver, FLAGS))
    return c.execute('SELECT id FROM key WHERE version = ? AND flags = ?', (ver, FLAGS)).fetchone()[0]

def source_id(c, path):
    c.execute('INSERT OR IGNORE INTO source(path, added) VALUES (?, ?)', (path, time.strftime('%Y-%m-%d %H:%M:%S')))
    return c.execute('SELECT id FROM source WHERE path = ?', (path,)).fetchone()[0]

def split_answer(rest):
    t, _, a = rest.partition('\t@')
    return t, int(a, 16) if a else -1

def fmt(w, t, a): return f'{w:016x}\t{t}' + (f'\t@{a:x}' if WANT_ADDR else '') + '\n'

def insert(c, k, src, answers):
    c.executemany('INSERT OR IGNORE INTO answer(key, word, addr, text, source) VALUES (?, ?, ?, ?, ?)',
                  ((k, s64(w), a, t, src) for w, a, t in answers))

# ---------------------------------------------------------------- live nvdisasm (recheck)
def nvmod():
    os.environ['NVADDR'] = '1'
    if not os.environ.get('NVTMP'): os.environ['NVTMP'] = tempfile.mkdtemp(prefix='nv-recheck.')
    spec = importlib.util.spec_from_file_location('nv', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nv.py'))
    m = importlib.util.module_from_spec(spec); sys.modules['nv'] = m; spec.loader.exec_module(m); return m

def live_batch(nv, words):
    import multiprocessing
    chunks = [words[i:i + 128] for i in range(0, len(words), 128)]
    out = {}
    with multiprocessing.get_context('fork').Pool(int(os.environ.get('NP', '3'))) as pool:
        for res in pool.imap_unordered(nv.chunk, chunks):
            for w, t in res.items(): out[w] = split_answer(t)
    return out

def live_at(nv, w, addr):
    """w asked alone at addr (NOPs in front); (text, addr)"""
    k = (addr - 8) // 32; NOP = nv.NOP
    res = nv.chunk([NOP] * k + [w])
    return split_answer(res.get(w, '!NOOUT'))

def sample_size(n): return min(n, max(200, math.ceil(n / 100)))

def recheck(out, label):
    """asks OUT.check (word<TAB>text<TAB>@addr, addr -1 when unknown) live; exit 3 on a mismatch"""
    sample = []
    with open(out + '.check') as f:
        seed = f.readline().split('\t')
        for l in f:
            w, t, a = l.rstrip('\n').split('\t'); sample.append((int(w, 16), int(a[1:]), t))
    os.unlink(out + '.check')
    hits, seed = int(seed[1]), int(seed[2])
    if not sample: return 0
    nv = nvmod()
    got = live_batch(nv, [w for w, _, _ in sample])
    bad, dep = [], 0
    for w, a, t in sample:
        lt, la = got.get(w, ('!NOOUT', -1))
        if lt == t: continue
        if a >= 0:
            lt, la = live_at(nv, w, a)
            if lt != t: bad.append((w, a, t, lt, la))
            continue
        lt2, la2 = live_at(nv, w, 8 if la != 8 else 40)
        if lt2 != lt: dep += 1  # address-dependent text, address unknown: not comparable
        else: bad.append((w, a, t, lt, la))
    import shutil; shutil.rmtree(os.environ['NVTMP'], ignore_errors=True)
    print(f'store recheck {label}: {len(sample)} of {hits} hits asked live (seed {seed}), {len(bad)} mismatches, '
          f'{dep} address-dependent with unknown address', file=sys.stderr)
    if bad:
        for w, a, t, lt, la in bad[:20]:
            print(f'STORE MISMATCH key=({version()!r}, flags, word={w:016x}, addr={a}) stored={t!r} live={lt!r} at {la:#x}', file=sys.stderr)
        print('STORE MISMATCH: the key is missing something that changes the text; fix the key, do not trust these hits', file=sys.stderr)
        sys.exit(3)
    return len(sample)

# ---------------------------------------------------------------- commands
def lookup(words_tsv, out):
    c = db(); k = key_id(c, version()); c.commit()
    lines = {}
    for l in open(words_tsv):
        if l.strip(): lines.setdefault(int(l.split('\t')[2], 16), l)
    c.execute('CREATE TEMP TABLE q(word INTEGER PRIMARY KEY)')
    c.executemany('INSERT INTO q VALUES (?)', ((s64(w),) for w in lines))
    hit = {}
    for w, a, t in c.execute('SELECT a.word, a.addr, a.text FROM q JOIN answer a ON a.key = ? AND a.word = q.word'
                             + (' AND a.addr >= 0' if WANT_ADDR else '') + ' ORDER BY a.rowid', (k,)):
        hit.setdefault(u64(w), (a, t))
    if os.environ.get('NVSTORE_LIVE') == '1': hit = {}  # a deliberate re-ask: nvdisasm answers all
    with open(out + '.hit', 'w') as o:
        for w, (a, t) in hit.items(): o.write(fmt(w, t, a))
    with open(words_tsv + '.ask', 'w') as o:
        for w, l in lines.items():
            if w not in hit: o.write(l)
    seed = int.from_bytes(os.urandom(4), 'little')
    sample = random.Random(seed).sample(sorted(hit), sample_size(len(hit))) if hit else []
    with open(out + '.check', 'w') as o:
        o.write(f'#hits\t{len(hit)}\t{seed}\n')
        for w in sample: o.write(f'{w:016x}\t{hit[w][1]}\t@{hit[w][0]}\n')
    label = os.environ.get('NVSTORE_LABEL') or os.path.basename(words_tsv)
    print(f'store {label}: {len(lines)} words, {len(hit)} answered by the store ({len(hit) / max(1, len(lines)):.1%}), '
          f'{len(lines) - len(hit)} to ask', file=sys.stderr)
    with open(STORE + '/take.log', 'a') as o:
        o.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{os.path.abspath(words_tsv)}\t{len(lines)}\t{len(hit)}\t{len(sample)}\n")

def take(words_tsv, out):
    lookup(words_tsv, out); recheck(out, os.path.basename(words_tsv))

def insert_file(out, source):
    c = db(); k = key_id(c, version()); src = source_id(c, source)
    new = []
    if os.path.exists(out):
        for l in open(out):
            w, _, rest = l.rstrip('\n').partition('\t')
            if w: t, a = split_answer(rest); new.append((int(w, 16), a, t))
    insert(c, k, src, new); c.commit()
    print(f"store {os.environ.get('NVSTORE_LABEL') or os.path.basename(out)}: {len(new)} new answers stored", file=sys.stderr)

def assemble(words_tsv, out):
    with open(out + '.part', 'w') as o:
        for l in open(out + '.hit'): o.write(l)
        if os.path.exists(out):
            for l in open(out):
                w, _, rest = l.rstrip('\n').partition('\t')
                if w: t, a = split_answer(rest); o.write(fmt(int(w, 16), t, a))
    os.replace(out + '.part', out); os.unlink(out + '.hit'); os.unlink(words_tsv + '.ask')

def put(words_tsv, out, source):
    insert_file(out, source); assemble(words_tsv, out)

def seed(nv_tsv, accepted=False):
    c = db(); k = key_id(c, version()); src = source_id(c, os.path.abspath(nv_tsv))
    ans, skipped = [], 0
    for l in open(nv_tsv):
        w, _, rest = l.rstrip('\n').partition('\t')
        if not w: continue
        t, a = split_answer(rest)
        if accepted and t.startswith('!'): skipped += 1; continue
        ans.append((int(w, 16), a, t))
    before = c.execute('SELECT count(*) FROM answer').fetchone()[0]
    insert(c, k, src, ans); c.commit()
    after = c.execute('SELECT count(*) FROM answer').fetchone()[0]
    print(f'seed {nv_tsv}: {len(ans)} answers, {after - before} new, {skipped} refusals skipped; store holds {after}', file=sys.stderr)

if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'take': take(a[0], a[1])
    elif cmd == 'put': put(a[0], a[1], a[2])
    elif cmd == 'lookup': lookup(a[0], a[1])
    elif cmd == 'recheck': recheck(a[0], a[1])
    elif cmd == 'insert': insert_file(a[0], a[1])
    elif cmd == 'assemble': assemble(a[0], a[1])
    elif cmd == 'seed': seed(a[0], len(a) > 1 and a[1] == 'accepted')
    else: sys.exit(f'unknown command {cmd}')
