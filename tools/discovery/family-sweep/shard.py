"""Split the words into per-host shards, balanced by expected cost.

Cost of a row piece = words * (1 + 300 * reject_rate), because each
rejected word costs a whole nvdisasm process (~300 batched words' worth).
reject_rate comes from the pilot run (64 words per row). Rows are cut
into pieces of <= 8192 words so a reject-heavy texture row can spread.
usage: shard.py WORDS_TSV PILOT_WORDS PILOT_NV OUTDIR host:workers ...
"""
import sys, collections
words, pw, pn, od = sys.argv[1:5]
# host:workers:speed; speed = per-worker nvdisasm throughput relative to
# the local machine (1.0), from the hosts file (see hosts.sh).
hosts = {h.split(':')[0]: int(h.split(':')[1]) * float(h.split(':')[2]) for h in sys.argv[5:]}
rej = {l.split('\t')[0]: l.split('\t')[1].startswith('!') for l in open(pn)}
rate = collections.defaultdict(lambda: [0, 0])
for l in open(pw):
    f = l.split('\t'); rate[f[0]][0] += 1; rate[f[0]][1] += rej.get(f[2], False)
rows = collections.defaultdict(list)
for l in open(words): rows[l.split('\t')[0]].append(l)
pieces = []
for k, ls in rows.items():
    r = rate[k][1] / rate[k][0] if rate[k][0] else 0.05
    for i in range(0, len(ls), 8192):
        p = ls[i:i + 8192]; pieces.append((len(p) * (1 + 300 * r), p))
load = {h: 0.0 for h in hosts}; files = {h: open(f'{od}/shard.{h}.tsv', 'w') for h in hosts}
for c, p in sorted(pieces, key=lambda x: -x[0]):
    h = min(hosts, key=lambda h: (load[h] + c) / hosts[h]); load[h] += c; files[h].writelines(p)
for h in hosts: print(f'shard {h}: workers {hosts[h]}, est cost {load[h]/hosts[h]:.0f}/effective worker', file=sys.stderr)
