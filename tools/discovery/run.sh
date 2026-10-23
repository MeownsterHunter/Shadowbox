#!/usr/bin/env bash
# run.sh - the whole decoder-discovery pipeline on one work dir, end to end.
#
# THE IDEA
#   One command reproduces docs/decoder-methodology.md methods 2 and 3:
#   synthesize instruction words, ask nvdisasm (the oracle) what each means,
#   compare with Shadowbox's decoder, and infer each family's bit layout.
#   Every input is Shadowbox's own decoder tables (crates/shadowbox/tests/
#   data/) plus synthetic words. run.sh is the only entry point anyone needs;
#   the step scripts it drives can also be run alone (each has --help).
#
# STEPS (each in its own subdirectory of the work dir)
#   sweep     family-sweep.sh, row mode: combinations inside known rows.
#             sweep/report.txt
#   opcode    family-sweep.sh --space opcode: all 65,536 opcode prefixes.
#             opcode/report.txt  (--no-opcode skips it)
#   infer     family-infer.sh: givens (manifest samples plus the opcode
#             run's accepted words), bit flips, pair flips, then layout and
#             validate.  infer/pairs.nv.tsv marks the probes done.
#   validate  family-infer.sh --from layout: layout, value tables, composed
#             words, validation against nvdisasm.  infer/validate.tsv
#
# RESUME
#   A step whose outputs exist in the work dir, made with the same step
#   flags (STEP/.runsig), is skipped. A step that runs makes every later
#   step run too. --from STEP reruns STEP and everything after it, against
#   the earlier steps' outputs: after a decoder or tool change, --from infer
#   redoes inference on the existing sweeps; --from validate only redoes
#   layout and validation. sweep, opcode and infer start from an empty
#   subdirectory when they run; validate keeps infer's probes.
#
# USAGE
#   run.sh [--work DIR] [--hosts-file FILE] [--hosts SEL] [--from STEP]
#          [--rows all|LINE,..] [--depth N] [--samples N] [--fills N]
#          [--no-opcode] [--repo DIR] [--store DIR|HOST:DIR|none]
#     --work        work dir (default $DISCOVERY_WORK, else
#                   <repo>/target/discovery); sbdecode builds in WORK/target
#                   unless CARGO_TARGET_DIR is set.
#     --hosts-file  optional hosts file (hosts.conf.example); default local only
#     --hosts       hosts.sh selector (default local)
#     --from        sweep | opcode | infer | validate
#     --rows        manifest line numbers for sweep and infer (default all)
#     --depth --samples  row-sweep size (defaults 18, 65536)
#     --fills       opcode-space random fills per prefix (default 2)
#     --no-opcode   skip the opcode-space run; inference uses samples only
#     --repo        shadowbox checkout (default: the one holding this script)
#     --store       the nvdisasm answer store ($DISCOVERY_STORE; default the
#                   hosts file's store= host, else WORK/store here; none
#                   asks every word and keeps nothing). It also holds the
#                   frozen evaluation set (STORE/frozen-eval).
#   MANY HOSTS  --hosts a,b,c (or all) runs every step on those hosts at
#     once: nvdisasm asks and the infer layout passes are work queues that
#     each host draws from as it frees up (family-infer.sh MANY HOSTS), all
#     sharing one answer store.
#   ENVIRONMENT  NVDISASM (default nvdisasm on PATH), DISCOVERY_WORKERS,
#     INFER_PER_ROW, INFER_MEMORY_MAX: see family-sweep.sh / family-infer.sh.
#
# OUTPUT
#   WORK/run.json  source commit, flags, environment (paths shown relative
#                  to the work dir or the repo, others redacted), nvdisasm
#                  version, per-step status and seconds.
#   stdout ends with a summary: sweep buckets, opcode-space counts,
#   validation rates (per-run composed set and the frozen set), what each
#   host did, and the report paths.
set -euo pipefail
TOOLS=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$TOOLS/../.." && pwd)
WORK= HOSTS_FILE= HOSTS=local STORE=${DISCOVERY_STORE:-} FROM= ROWS=all DEPTH=18 SAMPLES=65536 FILLS=2 OPCODE=1
while [ $# -gt 0 ]; do
  case $1 in
    --work) WORK=$2; shift 2;; --hosts-file) HOSTS_FILE=$2; shift 2;; --hosts) HOSTS=$2; shift 2;;
    --from) FROM=$2; shift 2;; --rows) ROWS=$2; shift 2;; --depth) DEPTH=$2; shift 2;;
    --samples) SAMPLES=$2; shift 2;; --fills) FILLS=$2; shift 2;; --no-opcode) OPCODE=0; shift;;
    --repo) REPO=$(cd "$2" && pwd); shift 2;; --store) STORE=$2; shift 2;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
STEPS=(sweep opcode infer validate)
case " ${STEPS[*]} " in *" ${FROM:-sweep} "*) ;; *) echo "unknown --from $FROM (${STEPS[*]})" >&2; exit 2;; esac
WORK=${WORK:-${DISCOVERY_WORK:-$REPO/target/discovery}}
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
export DISCOVERY_WORK=$WORK FS_HOST=$HOSTS
[ -n "$STORE" ] && export DISCOVERY_STORE=$STORE
[ -n "$HOSTS_FILE" ] && { HOSTS_FILE=$(cd "$(dirname "$HOSTS_FILE")" && pwd)/$(basename "$HOSTS_FILE"); export DISCOVERY_HOSTS=$HOSTS_FILE; }
ND=$(command -v "${NVDISASM:-nvdisasm}" || true)
[ -x "$ND" ] || { echo "nvdisasm not found: install NVIDIA's CUDA toolkit, then put nvdisasm on PATH or set NVDISASM" >&2; exit 1; }
export NVDISASM=$ND
T0=$(date +%s)
log() { echo "[run $(($(date +%s)-T0))s] $*" >&2; }

declare -A SIG=(
  [sweep]="rows=$ROWS depth=$DEPTH samples=$SAMPLES"
  [opcode]="fills=$FILLS"
  [infer]="rows=$ROWS oprun=$([ $OPCODE = 1 ] && echo opcode || echo none)"
  [validate]="rows=$ROWS oprun=$([ $OPCODE = 1 ] && echo opcode || echo none) per_row=${INFER_PER_ROW:-16}"
)
declare -A MARK=([sweep]=sweep/report.txt [opcode]=opcode/report.txt [infer]=infer/pairs.nv.tsv [validate]=infer/validate.tsv)
declare -A SIGFILE=([sweep]=sweep/.runsig [opcode]=opcode/.runsig [infer]=infer/.runsig [validate]=infer/.validatesig)
declare -A STATUS=() SECS=()
done_already() { # STEP: outputs present and made with the same step flags
  [ -s "$WORK/${MARK[$1]}" ] && [ "$(cat "$WORK/${SIGFILE[$1]}" 2>/dev/null)" = "${SIG[$1]}" ]
}
run_step() { # STEP: run it, time it, stamp its flags
  local s=$1 t=$(date +%s)
  case $s in
    sweep|opcode|infer) rm -rf "${WORK:?}/$s"; mkdir -p "$WORK/$s";;
  esac
  log "step $s (${SIG[$s]})"
  case $s in
    sweep) "$TOOLS/family-sweep.sh" --repo "$REPO" --hosts "$HOSTS" --rows "$ROWS" --depth "$DEPTH" \
             --samples "$SAMPLES" --out "$WORK/sweep" > /dev/null;;
    opcode) "$TOOLS/family-sweep.sh" --space opcode --repo "$REPO" --hosts "$HOSTS" --fills "$FILLS" \
             --out "$WORK/opcode" > /dev/null;;
    infer) "$TOOLS/family-infer.sh" --repo "$REPO" --hosts "$HOSTS" --rows "$ROWS" \
             --oprun "$([ $OPCODE = 1 ] && echo "$WORK/opcode" || echo none)" --out "$WORK/infer";;
    validate) "$TOOLS/family-infer.sh" --from layout --repo "$REPO" --hosts "$HOSTS" --rows "$ROWS" \
             --out "$WORK/infer";;
  esac
  echo "${SIG[$s]}" > "$WORK/${SIGFILE[$s]}"
  [ "$s" = infer ] && echo "${SIG[validate]}" > "$WORK/${SIGFILE[validate]}"
  STATUS[$s]=ran; SECS[$s]=$(($(date +%s)-t))
}
FORCE=0
for s in "${STEPS[@]}"; do
  [ "$s" = "${FROM:-}" ] && FORCE=1
  if [ "$s" = opcode ] && [ $OPCODE = 0 ]; then STATUS[$s]=off; continue; fi
  if [ "$s" = validate ] && [ "${STATUS[infer]:-}" = ran ]; then STATUS[$s]=ran-with-infer; continue; fi
  if [ $FORCE = 0 ] && done_already "$s"; then STATUS[$s]=skipped; log "step $s: outputs present, skipped"; continue; fi
  FORCE=1; run_step "$s"
done

# run.json: everything needed to say what produced this work dir.
for s in "${STEPS[@]}"; do echo "$s ${STATUS[$s]} ${SECS[$s]:-}"; done > "$WORK/.steps"
RUN_STORE=$STORE RUN_REPO=$REPO RUN_WORK=$WORK RUN_TOOLS=$TOOLS RUN_ND=$ND RUN_HOSTS_FILE=$HOSTS_FILE \
RUN_FLAGS="from=${FROM:-} hosts=$HOSTS rows=$ROWS depth=$DEPTH samples=$SAMPLES fills=$FILLS opcode=$OPCODE" \
RUN_SECS=$(($(date +%s)-T0)) python3 -I - <<'PY'
import json, os, subprocess
E = os.environ; work = E['RUN_WORK']; repo = E['RUN_REPO']
def git(*a):
    r = subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None
def shown(p):  # a path relative to the work dir or repo; anything else redacted
    if not p: return None
    for root, tag in ((work, 'WORK'), (repo, 'REPO')):
        if p == root or p.startswith(root + '/'): return tag + p[len(root):]
    return '<redacted>/' + os.path.basename(p)
v = subprocess.run([E['RUN_ND'], '--version'], capture_output=True, text=True).stdout.strip().splitlines()
prev = {}
try: prev = {s['step']: s for s in json.load(open(work + '/run.json'))['steps']}
except Exception: pass
steps = []
for l in open(work + '/.steps'):
    f = l.split() + ['']
    s = dict(step=f[0], status=f[1], seconds=int(f[2]) if f[2] else None)
    if s['status'] == 'skipped' and f[0] in prev:
        s['seconds_when_made'] = prev[f[0]].get('seconds') or prev[f[0]].get('seconds_when_made')
    steps.append(s)
os.unlink(work + '/.steps')
env = {k: (shown(E[k]) if k in ('CARGO_TARGET_DIR',) else E[k])
       for k in ('DISCOVERY_WORKERS', 'INFER_PER_ROW', 'INFER_MEMORY_MAX', 'CARGO_TARGET_DIR') if E.get(k)}
env['NVDISASM'] = shown(E['RUN_ND'])
env['DISCOVERY_HOSTS'] = shown(E['RUN_HOSTS_FILE']) if E.get('RUN_HOSTS_FILE') else None
st = E.get('RUN_STORE', '')
env['DISCOVERY_STORE'] = st if st in ('', 'none') else (shown(st) if st.startswith('/') else st.split(':', 1)[0] + ':<redacted>' if ':' in st else '<redacted>')
json.dump(dict(
    source_commit=git('rev-parse', 'HEAD'),
    source_dirty=bool(git('status', '--porcelain', '--', 'tools/discovery', 'crates/shadowbox')),
    tools=shown(E['RUN_TOOLS']),
    flags=dict(kv.split('=', 1) for kv in E['RUN_FLAGS'].split()),
    env=env, nvdisasm_version=v[-2] if len(v) >= 2 else (v[-1] if v else None),
    steps=steps, seconds=int(E['RUN_SECS'])), open(work + '/run.json', 'w'), indent=1)
PY

# Summary: coverage numbers and where the reports are.
echo "== discovery run: $WORK (run.json)"
if [ -s "$WORK/sweep/report.txt" ]; then
  echo "-- row sweep: $WORK/sweep/report.txt (clusters.tsv, wrong_admissions.tsv)"
  sed -n '1,2p' "$WORK/sweep/report.txt" | sed 's/^/   /'
fi
if [ -s "$WORK/opcode/report.txt" ]; then
  echo "-- opcode space: $WORK/opcode/report.txt (new_families.tsv, new_prefixes.tsv)"
  sed -n '1p' "$WORK/opcode/report.txt" | sed 's/^/   /'
fi
if [ -s "$WORK/infer/validate.tsv" ]; then
  echo "-- inference: $WORK/infer/layouts.tsv, conflicts.tsv, validate.tsv"
  awk -F'\t' '!/^#/ { r++; n += $2; e += $3; c += $4; j += $5; if ($2 == $3) full++ }
    END { if (n) printf "   rows %d  composed words %d  exact %d (%.1f%%)  canonical %d (%.1f%%)  rejected %d  rows all-exact %d\n",
          r, n, e, 100 * e / n, c, 100 * c / n, j, full }' "$WORK/infer/validate.tsv"
  awk -F'\t' '!/^#/ && !r[$1]++ { n++ } END { printf "   layouts for %d rows\n", n }' "$WORK/infer/layouts.tsv"
fi
if [ -s "$WORK/infer/frozen/validate.tsv" ]; then
  echo "-- frozen evaluation set: $WORK/infer/frozen/validate.tsv"
  awk -F'\t' '!/^#/ { r++; n += $2; e += $3; c += $4; j += $5; if ($2 == $3) full++ }
    END { if (n) printf "   rows %d  frozen words %d  exact %d (%.1f%%)  canonical %d (%.1f%%)  rejected %d  rows all-exact %d\n",
          r, n, e, 100 * e / n, c, 100 * c / n, j, full }' "$WORK/infer/frozen/validate.tsv"
fi
if [ -s "$WORK/infer/shares.tsv" ]; then
  echo "-- hosts (infer/shares.tsv): per host, nvdisasm words asked and layout units run"
  awk -F'\t' '$1 == "nv" { nw[$3] += $5; h[$3] = 1 } $1 == "unit" { u[$3]++; r[$3] += $4; b[$3] += $5; s[$3] += $7; h[$3] = 1 }
    END { for (x in h) printf "   %-12s nvdisasm words %d  units %d  rows %d  bases %d  busy %ds\n", x, nw[x], u[x], r[x], b[x], s[x] }' \
    "$WORK/infer/shares.tsv" | sort
fi
echo "-- steps: $(for s in "${STEPS[@]}"; do printf '%s=%s ' "$s" "${STATUS[$s]}"; done)total $(($(date +%s)-T0))s"
