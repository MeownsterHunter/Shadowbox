#!/usr/bin/env bash
# family-sweep.sh - find Maxwell (SM52) instruction words that NVIDIA's
# disassembler accepts but Shadowbox's decoder calls Unknown, and the reverse.
#
# THE IDEA
#   docs/decoder-coverage.md samples the envytools gm107 table one field at a
#   time. That proves each field value alone, never combinations, so real
#   shaders keep hitting family members nobody probed (F2F with an F16
#   high-half source: row 0x5ca8 with bit 41; IPA reading attribute 0x3fc
#   FrontFacing). nvdisasm is the oracle: if it prints an instruction and
#   shadowbox says Unknown, that is an undiscovered member; if shadowbox
#   decodes a word nvdisasm rejects, that is a wrong admission; if both decode
#   but the shadowbox kind cannot stand for that mnemonic, that is a name
#   mismatch.
#
# THE ALGORITHM (helpers in family-sweep/)
#   1 fields.py   For each manifest row (crates/shadowbox/tests/data/
#                 decoder-rows.tsv) derive field bit masks from decoder-
#                 samples.tsv; split bits into modifier, data (operands) and
#                 uncovered (no sampled field touches them).
#   2 anchors.py  Pick operand values shadowbox can decode for the row
#                 (registers RZ or baseline, predicates PT or baseline,
#                 attribute 0x80 or baseline); the manifest baseline uses all
#                 zeros, which hides whole rows (IPA needs RZ/PT).
#   3 gen.py      Opcode fixed, operands at the anchor (and the baseline);
#                 modifier bits exhaustive up to DEPTH bits, else SAMPLES
#                 seeded-random words; plus uncovered-bit twins and pairs,
#                 every attribute address for AST/ALD/IPA, every system
#                 register for S2R/CS2R.
#   4 nv.py       nvdisasm batches, sharded across hosts by shard.py.
#   5 sbdecode    Tiny binary over shadowbox::decoder::decode_word, built from
#                 the checkout into $CARGO_TARGET_DIR; prints the kind.
#   6 analyze.py  Buckets, clusters by mnemonic+modifiers+operand shape,
#                 graphics-weighted rank, wrong-admission bit signatures,
#                 attribute-address gaps; writes report.txt and TSVs.
#   Every input is Shadowbox's own decoder tables plus synthetic words.
#
# NVDISASM BATCH FORMAT
#   Raw little-endian 64-bit words in groups of four:
#     [CONTROL 0x001fc400fe2007f6, WORD, NOP 0x50b0000000070f00, NOP]
#   run as `nvdisasm -b SM52 -raw FILE`; WORD is at byte 32*i+8. nvdisasm
#   aborts the whole file at the first undecodable word and names its address,
#   so nv.py marks that word rejected and reruns the remainder: one process per
#   128-word chunk plus one per rejected word. Rejects dominate the cost
#   (texture rows reject ~half their words).
#
# ANSWER STORE (family-sweep/store.py has the format and keying)
#   Every nvdisasm answer ever made is kept in one insert-only store (where:
#   hosts.sh ANSWER STORE), keyed by (nvdisasm binary digest, the invocation
#   and batch format above, word) with the address nvdisasm printed the word
#   at. The pilot and the sweep first take the store's answers, ask nvdisasm
#   only for the rest (asked with NVADDR=1 so the address is kept) and store
#   the new ones; nv.tsv and pilot.nv.tsv come out as before. A sample of the
#   hits (1%, at least 200) is asked live every run; a mismatch fails the run
#   and names the key. DISCOVERY_STORE=none turns the store off.
#
# HOSTS (hosts.sh; all work at nice 19; nothing touches a GPU)
#   Local only unless you write a hosts file (hosts.conf.example). --hosts
#   (or FS_HOST) takes a hosts.sh selector: a host, a list (local,box2), auto
#   (the freest host) or all (every host with free cores, each at its free
#   capacity: nproc minus loadavg). Shards are balanced on workers x speed.
#   Words are generated here, shipped as per-host shards (row pieces
#   balanced by a pilot reject rate), and only nvdisasm text comes back. The
#   pilot and Shadowbox classification always run here, against --repo.
#   Scale: the full default sweep is about 1.9M words, dominated by nvdisasm
#   time; use --rows, --depth and --samples to make a small run.
#
# OPCODE-SPACE MODE (--space opcode)
#   THE IDEA  The row mode above fixes opcode bits to manifest rows, so it
#   fills in around what we know and is blind to any family whose opcode
#   pattern is not a manifest row. Opcode mode enumerates the opcode space
#   itself: every value of bits 63..48 (the granularity of row keys like
#   0x5ca8), 65,536 prefixes, each at neutral operand anchors
#   (gen_opcode.py): all-zero low 48 bits; RZ (0xff) in Rd/Ra/Rb/Rc with PT
#   in the guard (the RZ bytes also put PT in the predicate fields); and
#   --fills (default 2) seeded-random low 48 bits with a PT guard. nvdisasm
#   and sbdecode classify every word; no fields/anchors/pilot step.
#   OUTPUT (analyze_opcode.py; stdout = report.txt)
#     new_prefixes.tsv   prefixes nvdisasm decodes that no root manifest row
#                        covers on its top-16 mask bits: undiscovered
#                        families; mnemonic, shadowbox kinds, example word.
#     new_families.tsv   the same clustered by base mnemonic.
#     unknown_known.tsv  decodes, shadowbox Unknown, a manifest row matches.
#     row_gap.tsv        decodes, prefix covered, but no row matches the word.
#     wrong_admissions.tsv  nvdisasm rejects, shadowbox decodes.
#     split.tsv          prefixes whose words disagree on the base mnemonic:
#                        opcode bits below 48; noted for a follow-up split,
#                        never recursed automatically.
#     words.tsv          prefix, effective row, word, anchor, prefix_row.
#   COST  4 words x 65,536 = 262,144 words; most prefixes reject, and each
#   rejected word is one nvdisasm process, so this is reject-bound. The words
#   go to every selected host as a work queue (hosts.sh host_queue_nv).
#   RESWEEP  --prefixes FILE (one hex prefix per line, first column) sweeps
#   only those prefixes, with --fills random fills each drawn from --seed:
#   more samples where an earlier sweep found nothing it could cover.
#   USAGE  family-sweep.sh --space opcode [--fills N] [--prefixes FILE --seed S]
#          [--hosts SEL] [--repo DIR] [--out DIR]   (--rows/--depth/--samples ignored)
#
# USAGE
#   family-sweep.sh [--space rows|opcode] [--rows all|LINE,LINE..] [--depth N] [--samples N]
#                   [--hosts SEL] [--repo DIR] [--out DIR]
#     --rows     manifest line numbers (gm107.c:LINE), default all
#     --depth    rows with <= N modifier bits are swept exhaustively (default 18)
#     --samples  random words for rows above depth (default 65536)
#     --hosts    hosts.sh selector: local, a host from the hosts file, a comma
#                list, auto or all (default $FS_HOST, else local)
#     --repo     shadowbox checkout, read-only (default: the checkout holding
#                this script)
#     --out      run dir (default $DISCOVERY_WORK/runs/<time>)
#   ENVIRONMENT
#     DISCOVERY_WORK    work root (default <repo>/target/discovery): run dirs
#                       and the sbdecode build dir.
#     CARGO_TARGET_DIR  where sbdecode builds (default $DISCOVERY_WORK/target).
#     NVDISASM          nvdisasm binary (default nvdisasm on PATH).
#     DISCOVERY_HOSTS   hosts file (default tools/discovery/hosts.conf).
#     DISCOVERY_WORKERS local nvdisasm workers (default the CPU count).
#
# OUTPUT (stdout = report.txt; data in the run dir)
#   BUCKETS        missing / wrong-admission / name-mismatch word counts.
#   clusters.tsv   one line per missing cluster, ranked. Categories:
#                  new-operands, new-modifiers (real family gaps, weight 3),
#                  operand-values (only register/predicate/attribute values
#                  differ from a decoded word), same-text (identical
#                  disassembly decodes elsewhere: decoder rejects bits nvdisasm
#                  ignores), row-unimplemented (row decodes nothing).
#                  'diff' names each bit that separates the example from the
#                  nearest decoded word in the same row, with its field.
#   wrong_admissions.tsv  grouped by row/kind/error, with bits whose value
#                  predicts rejection among that row's decoded words.
#   mismatch.tsv   kind/mnemonic pairs outside analyze.py KIND_OPS.
#   attributes.tsv attribute addresses nvdisasm accepts that never decode.
#   words.tsv nv.tsv sb.tsv  raw: row, effective row, word, anchor tag /
#                  nvdisasm text or !REJECT / shadowbox kind.
# Read-only for the checkout; writes only the run dir, the sbdecode build dir
# and the remote work dirs. Never commits, never kills anything.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)/family-sweep
SPACE=rows FILLS=2 PREFIXES= SEED=5253 ROWS=all DEPTH=18 SAMPLES=65536 HOSTS=${FS_HOST:-local}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=
while [ $# -gt 0 ]; do
  case $1 in
    --rows) ROWS=$2; shift 2;; --depth) DEPTH=$2; shift 2;; --samples) SAMPLES=$2; shift 2;;
    --space) SPACE=$2; shift 2;; --fills) FILLS=$2; shift 2;;
    --prefixes) PREFIXES=$2; shift 2;; --seed) SEED=$2; shift 2;;
    --hosts) HOSTS=$2; shift 2;; --repo) REPO=$2; shift 2;; --out) OUT=$2; shift 2;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
BASE=${DISCOVERY_WORK:-$REPO/target/discovery}
OUT=${OUT:-$BASE/runs/$(date +%Y%m%d-%H%M%S)}
. "$(dirname "$HERE")/hosts.sh"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$BASE/target}
mkdir -p "$OUT"; RUNID=$(basename "$OUT"); T0=$(date +%s)
log() { echo "[$(($(date +%s)-T0))s] $*" >&2; }

log "build sbdecode against $REPO"
B=$BASE/sbdecode-build; mkdir -p "$B/src"
cp "$HERE/sbdecode.rs" "$B/src/main.rs"; cp "$REPO/Cargo.lock" "$B/Cargo.lock"
cat > "$B/Cargo.toml" <<TOML
[package]
name = "sbdecode"
version = "0.1.0"
edition = "2021"
publish = false
[dependencies]
shadowbox = { path = "$REPO/crates/shadowbox" }
[workspace]
TOML
(cd "$B" && nice -n 19 cargo build --release -q 2>&1 | rg -v '^warning|^\s|^$' >&2 || true)
SB=$CARGO_TARGET_DIR/release/sbdecode; [ -x "$SB" ] || { echo "sbdecode build failed" >&2; exit 1; }

DATA=$REPO/crates/shadowbox/tests/data
classify() { # shadowbox kind for every words.tsv word -> sb.tsv
  python3 -I -c "
import struct,sys
o=open(sys.argv[2],'wb')
for l in open(sys.argv[1]): o.write(struct.pack('<Q',int(l.split('\t')[2],16)))" "$OUT/words.tsv" "$OUT/words.bin"
  "$SB" < "$OUT/words.bin" > "$OUT/sb.tsv"; rm -f "$OUT/words.bin"
}
run_host() { # nvdisasm over host $1's shard; only words go, only text comes back
  NVADDR=1 host_nv "$1" "$OUT/shard.$1.tsv" "$OUT/nv.$1.tsv" "$RUNID" 128 && log "host $1 done ($(wc -l < "$OUT/nv.$1.tsv") words)"
}
if [ "$SPACE" = opcode ]; then
  log "generate opcode-space words (fills=$FILLS)"
  nice -n 19 python3 -I "$HERE/gen_opcode.py" "$DATA/decoder-rows.tsv" "$FILLS" "$OUT/words.tsv" ${PREFIXES:+"$PREFIXES" "$SEED"}
  host_select "$HOSTS" local
  host_nvdisasm_check || exit 1
  nv_store_take "$OUT/words.tsv" "$OUT/nv.tsv" || exit 1
  log "nvdisasm sweep on ${HOSTS_SEL[*]}, as a work queue"
  if [ -s "$OUT/words.tsv.ask" ]; then
    NVADDR=1 host_queue_nv "$OUT/words.tsv.ask" "$OUT/nv.tsv" "$RUNID-opcode" 128 \
      || { echo "nvdisasm failed; partial results in $OUT" >&2; exit 1; }
  else : > "$OUT/nv.tsv"; fi
  nv_store_put "$OUT/words.tsv" "$OUT/nv.tsv" "$OUT/nv" || exit 1
  log "shadowbox classification"; classify
  log "analyze"
  ROWS_TSV=$DATA/decoder-rows.tsv nice -n 19 python3 -I "$HERE/analyze_opcode.py" "$OUT"
  echo "RUNTIME $(($(date +%s)-T0))s  command: $0 --space opcode --fills $FILLS${PREFIXES:+ --prefixes $PREFIXES --seed $SEED} --hosts $HOSTS --repo $REPO --out $OUT"
  exit 0
fi
[ "$SPACE" = rows ] || { echo "unknown --space $SPACE (rows|opcode)" >&2; exit 2; }
nice -n 19 python3 -I "$HERE/fields.py" "$DATA" "$OUT/rows.json"
nice -n 19 python3 -I "$HERE/anchors.py" "$OUT/rows.json" "$SB" "$OUT/anchors.json"
log "generate words (rows=$ROWS depth=$DEPTH samples=$SAMPLES)"
nice -n 19 python3 -I "$HERE/gen.py" "$OUT/rows.json" "$OUT/anchors.json" "$ROWS" "$DEPTH" "$SAMPLES" "$OUT/words.tsv"

log "pilot: 64 words per row, locally"
awk -F'\t' '{n[$1]++} n[$1]<=64' "$OUT/words.tsv" > "$OUT/pilot.words.tsv"
nv_store_take "$OUT/pilot.words.tsv" "$OUT/pilot.nv.tsv" || exit 1
HOST_CAP[local]=${HOST_WORKERS[local]}; NVADDR=1 host_nv local "$OUT/pilot.words.tsv.ask" "$OUT/pilot.nv.tsv" pilot 128
nv_store_put "$OUT/pilot.words.tsv" "$OUT/pilot.nv.tsv" "$OUT/pilot" || exit 1
host_select "$HOSTS" local; SPEC=()
for h in "${HOSTS_SEL[@]}"; do SPEC+=("$h:${HOST_CAP[$h]}:${HOST_SPEED[$h]}"); done
host_nvdisasm_check || exit 1
nv_store_take "$OUT/words.tsv" "$OUT/nv.tsv" || exit 1
python3 -I "$HERE/shard.py" "$OUT/words.tsv.ask" "$OUT/pilot.words.tsv" "$OUT/pilot.nv.tsv" "$OUT" "${SPEC[@]}"

log "nvdisasm sweep on ${HOSTS_SEL[*]}"
host_parallel run_host || { echo "a host failed; partial results in $OUT" >&2; exit 1; }
for h in "${HOSTS_SEL[@]}"; do cat "$OUT/nv.$h.tsv"; done | rg -v '^$' > "$OUT/nv.tsv" || true
nv_store_put "$OUT/words.tsv" "$OUT/nv.tsv" "$OUT/nv" || exit 1

log "shadowbox classification"; classify
log "analyze"
nice -n 19 python3 -I "$HERE/analyze.py" "$OUT/rows.json" "$OUT/anchors.json" "$OUT"
echo "RUNTIME $(($(date +%s)-T0))s  command: $0 --rows $ROWS --depth $DEPTH --samples $SAMPLES --hosts $HOSTS --repo $REPO --out $OUT"
