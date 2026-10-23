"""Ask nvdisasm about every word in a shard.

Batch format: a raw little-endian file of 32-byte groups
  [CONTROL 0x001fc400fe2007f6, WORD, NOP 0x50b0000000070f00, NOP]
run as `nvdisasm -b SM52 -raw file`. WORD sits at byte 32*i+8.
nvdisasm aborts the whole file at the first word it cannot decode and
names its address ("... at address 0x..."). So: run a chunk; on error
mark that word rejected (with the message), and rerun the rest of the
chunk. Cost is one process per chunk plus one per rejected word.
usage: ND=nvdisasm NVTMP=dir NP=workers CH=chunk nv.py IN_TSV OUT_TSV
  NP defaults to the CPU count, CH to 128 words per nvdisasm batch.
  IN_TSV column 3 is the word; OUT_TSV: word_hex<TAB>text or !REJECT msg
  NVADDR=1: each text is followed by <TAB>@address (hex) where nvdisasm
  placed the word, for texts that depend on it (PC-relative targets).
"""
import sys, struct, subprocess, re, os, tempfile
from multiprocessing import Pool
ND = os.environ['ND']; TMP = os.environ['NVTMP']; os.makedirs(TMP, exist_ok=True)
C = 0x001fc400fe2007f6; NOP = 0x50b0000000070f00
ADDR = os.environ.get('NVADDR') == '1'
ERR = re.compile(r'at address 0x([0-9a-fA-F]+)')
LINE = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;')
def run(words):
    fd, p = tempfile.mkstemp(dir=TMP, suffix='.bin')
    with os.fdopen(fd, 'wb') as f:
        for w in words: f.write(struct.pack('<4Q', C, w, NOP, NOP))
    r = subprocess.run([ND, '-b', 'SM52', '-raw', p], capture_output=True, text=True)
    os.unlink(p); return r
def parse(sub, text, limit, res):
    for m in LINE.finditer(text):
        a = int(m.group(1), 16)
        if a % 32 == 8 and a // 32 < limit: res[sub[a // 32]] = m.group(2) + (f'\t@{a:x}' if ADDR else '')
def chunk(words):
    res = {}; i = 0
    while i < len(words):
        sub = words[i:]; r = run(sub)
        if r.returncode == 0:
            parse(sub, r.stdout, len(sub), res)
            for w in sub: res.setdefault(w, '!NOOUT')
            break
        m = ERR.search(r.stderr)
        j = int(m.group(1), 16) // 32 if m else 0
        msg = r.stderr.strip().split('\n')[0].replace('\t', ' ')
        if not m and len(sub) > 1:
            r1 = run(sub[:1]); msg = r1.stderr.strip().split('\n')[0].replace('\t', ' ')
            if r1.returncode == 0:
                parse(sub[:1], r1.stdout, 1, res); i += 1; continue
        parse(sub, r.stdout, j, res)
        res[sub[j]] = '!REJECT ' + re.sub(r' at address 0x[0-9a-fA-F]+', '', msg)[:200]
        redo = [w for w in sub[:j] if w not in res]
        if redo: res.update(chunk(redo))
        i += j + 1
    return res
if __name__ == '__main__':
    words = [int(l.split('\t')[2], 16) for l in open(sys.argv[1])]
    CH = int(os.environ.get('CH', '128'))
    chunks = [words[i:i + CH] for i in range(0, len(words), CH)]
    with Pool(int(os.environ.get('NP') or os.cpu_count() or 1)) as pool, open(sys.argv[2] + '.part', 'w') as out:
        for res in pool.imap_unordered(chunk, chunks):
            for w, t in res.items(): out.write(f'{w:016x}\t{t}\n')
    os.replace(sys.argv[2] + '.part', sys.argv[2])
