#!/usr/bin/env bash
# family-infer.sh - infer each Maxwell (SM52) manifest row's bit layout from
# nvdisasm, like a sudoku: from known words, the role of every bit.
#
# THE IDEA
#   family-sweep.sh discovers members by sampling candidate words and asking
#   nvdisasm (the oracle). Sampling never ends: combinations outgrow it. Here
#   the layout itself is the result. Flip one bit of a word nvdisasm accepts
#   and diff the two texts: the change says what that bit is (opcode,
#   modifier, register field, immediate, or nothing). Merged over many words
#   of a row, the bits form fields with roles, widths and modifier value
#   tables, so members never seen decode by construction. The proof is
#   constructive too: compose new words from the layout, predict their text,
#   and ask nvdisasm whether it prints exactly that.
#
# THE ALGORITHM (family-sweep/infer.py; its docstring has the text model and
#   the class rules)
#   1 bases     Known-valid words per root manifest row: every sample of
#               decoder-samples.tsv nvdisasm accepts, plus the nvdisasm-valid
#               words of an opcode-space run (--oprun, optional) by
#               effective row. One
#               base per (row, mnemonic, modifiers, operand shape): another
#               word of the same shape only repeats the same flips.
#   2 flips     Each base with each of its 64 bits flipped, nvdisasm'd in
#               batches (nv.py) split over --hosts.
#   3 classify  Per flip: opcode (mnemonic changes or rejected), form
#               (operand count, atom count or atom kind: an immediate that
#               becomes a register is another operand form), mod
#               (dotted suffix appears/changes; names kept), enum, flag
#               (-, |..|, !, ~ on an operand), reg (an R/P number changes by
#               2^k: field start = bit-k), imm (a number changes; value bit,
#               sign or float), none.
#   4 pairs     Adjacent bits that are both modifier/enum/flag on a base are
#               flipped together: if the pair text is not the two single flips
#               combined, the bits are one field.
#   5 layout    Per row, per operand form: the bits whose flip changes the
#               operand list ("form" bits) select the form, and each value
#               of them seen on a base gets its own layout (texture, surface,
#               attribute and video rows move operands between forms).
#               Within a form: contiguous fields of one role and key.
#               Immediates: every run of one operand is one value, low run
#               first; a top bit that flips the sign or moves the value by
#               more than 2^k makes it two's complement over that width.
#               Modifier fields printing overlapping names are one field; a
#               modifier whose name still depends on another selector field
#               (pair flips and every accepted word are the evidence) joins
#               it: a joint field over both, named by their product.
#               Dependent rendering: an operand's shape (R.B1 vs R.H1) and an
#               immediate's signedness can be decided by selector values, not
#               by one field alone. Per operand, the selector components whose
#               joint value decides it are grown one at a time while that
#               removes disagreement over every accepted word of the form; the
#               table keeps the key values that name one result (layouts.tsv
#               render@slot rows, "sign by" on immediates). A modifier field's
#               value prints its names in the order seen for that value; only
#               separate fields are ordered by the row's pairwise counts.
#   5b tables   Every value (2^w) of every modifier/enum field is set on the
#               form's first base and nvdisasm names it (a reject = invalid
#               value), so tables hold all values, not just observed ones. A
#               joint field is the product of its components: each component
#               is enumerated alone first, then only the product of their
#               valid values is asked (sharded like any other batch).
#               Layout reruns until no new field needs enumerating.
#               Conflicts (a bit with different roles on different bases, a
#               role on some and no effect on others, disagreeing slots,
#               ambiguous modifier values) are the findings, not errors.
#   5c pc       Every base is asked again at a known address (nv.py with
#               NVADDR=1 prints it); a number that moved with the address is
#               a branch target: that row's words are asked again with
#               addresses and the number is learned relative to the word's
#               address (plus a learned constant); a prediction names it as
#               PC+value and validate resolves it with the word's address.
#   6 validate  Per row, INFER_PER_ROW (16) new words: every register,
#               immediate, modifier, flag and no-effect field set at random
#               (modifiers to values in their table) on a base, never a known
#               word; predicted text vs nvdisasm. exact = same string;
#               canonical = same mnemonic, modifier set and operands.
#               This per-run set follows the current layout, so it is redrawn
#               whenever the layout changes; it stays as the check against a
#               model fitted to the frozen set.
#   6b frozen   The same numbers on the frozen evaluation set
#               (<store>/frozen-eval, family-sweep/frozen.py): a fixed
#               held-out word list per row, drawn once with a recorded seed,
#               so every run's exact-match numbers compare (R/frozen/).
#
# ANSWER STORE (family-sweep/store.py has the format and keying)
#   Every nvdisasm ask (nv_step) first takes the answers the shared store
#   already holds for its words (key: nvdisasm binary digest, invocation and
#   batch format, word; the address the word was printed at is kept with the
#   text), asks nvdisasm only for the rest, and stores the new answers.
#   *.nv.tsv come out as before. A random 1% of the hits (at least 200) is
#   asked live every step; a mismatch fails the run and names the key. Steps
#   in NV_LIVE_STEPS re-ask known words on purpose and take nothing from the
#   store. Where the store lives: hosts.sh ANSWER STORE.
#
# MANY HOSTS AT ONCE
#   Every nvdisasm ask is a work queue over the selected hosts (hosts.sh
#   host_queue_nv): each host takes the next chunk of words when it finishes
#   one. The layout passes (5, 5b) are a second queue: layout works row by
#   row, so rows are packed into units (family-sweep/units.py) and every
#   host runs WORKERS units at a time, each to its own convergence, taking
#   the next unit (heaviest first) when one is done. The units' results are
#   merged into one layouts.tsv, conflicts.tsv and model.json, byte for byte
#   what one process over all rows writes; compose, validate and the frozen
#   set then run here. R/shares.tsv records which host did what.
#
# USAGE
#   family-infer.sh [--repo DIR] [--oprun DIR|none] [--rows all|LINE,LINE..]
#                   [--hosts SEL] [--caps HOST:N,..] [--out DIR] [--from start|layout]
#     --repo   shadowbox checkout, read-only (default: the checkout holding
#              this script)
#     --oprun  opcode-space run dir with words.tsv + nv.tsv, from
#              family-sweep.sh --space opcode (default none: the givens are
#              the manifest samples only)
#     --rows   manifest line numbers (gm107.c:LINE) to infer, default all
#     --hosts  hosts.sh selector for nvdisasm (default $FS_HOST, else local)
#     --caps   workers per host, overriding the hosts file
#     --out    run dir (default $DISCOVERY_WORK/runs/infer-<time>)
#     --from   start (default) or layout: reuse the run dir's samples, bases,
#              flips and pairs and redo layout, tables, compose, validate
#   ENVIRONMENT
#     DISCOVERY_WORK    work root (default <repo>/target/discovery).
#     NVDISASM, DISCOVERY_HOSTS, DISCOVERY_WORKERS  as in family-sweep.sh.
#     INFER_PER_ROW     composed validation words per row (default 16).
#     INFER_UNITS       layout units (default: twice the selected workers;
#                       0 runs the layout passes in one process here).
#     DISCOVERY_STORE   the answer store (hosts.sh ANSWER STORE).
#     INFER_MEMORY_MAX  optional memory cap such as 16G: when set and
#                       systemd-run exists, re-exec in a user scope with that
#                       MemoryMax, no swap, nice 19 (INFER_SCOPED marks it).
#   Scale: every root row is about 1.8M single flips; --rows makes it small.
#
# OUTPUT (run dir)
#   layouts.tsv    row, form (form-bit value, hex; - when none), field
#                  (role@operand slot, mod, mod(joint), opcode, none), bits
#                  hi:lo+hi:lo (joint/split fields, high part first), role, names (modifier value=suffixes table, enum
#                  value=operand text, register width, immediate scale,
#                  opcode transitions/invalid).
#   conflicts.tsv  row, form, bit, kind, class counts, example base word.
#   validate.tsv   per row: words, exact, canonical, rejected, rates.
#   validate-words.tsv  every composed word, predicted and printed text.
#   frozen/validate.tsv, frozen/validate-words.tsv  the same on the frozen set.
#   shares.tsv     per host: nvdisasm chunks and words; layout units, their
#                  rows, bases, passes and seconds.
#   bases.tsv model.json *.words.tsv *.nv.tsv  raw.
# Read-only for the checkout; writes only the run dir and remote work dirs. Never commits, never kills anything.
set -euo pipefail
TOOLS=$(cd "$(dirname "$0")" && pwd); HERE=$TOOLS/family-sweep
REPO=$(cd "$TOOLS/../.." && pwd) OPRUN=none ROWS=all
HOSTS=${FS_HOST:-local} CAPS= FROM=start OUT=
ARGS=("$@")
while [ $# -gt 0 ]; do
  case $1 in
    --repo) REPO=$2; shift 2;; --oprun) OPRUN=$2; shift 2;; --hosts) HOSTS=$2; shift 2;;
    --caps) CAPS=$2; shift 2;; --out) OUT=$2; shift 2;; --from) FROM=$2; shift 2;;
    --rows) ROWS=$2; shift 2;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
if [ -n "${INFER_MEMORY_MAX:-}" ] && [ -z "${INFER_SCOPED:-}" ] && command -v systemd-run > /dev/null; then
  INFER_SCOPED=1 exec systemd-run --user --scope -q -p MemoryMax="$INFER_MEMORY_MAX" -p MemorySwapMax=0 \
    --setenv=INFER_SCOPED=1 nice -n 19 "$0" "${ARGS[@]}"
fi
OUT=${OUT:-${DISCOVERY_WORK:-$REPO/target/discovery}/runs/infer-$(date +%Y%m%d-%H%M%S)}
. "$TOOLS/hosts.sh"
export NVADDR=1  # nvdisasm texts (and store answers) carry the address the word sat at
mkdir -p "$OUT"; RUNID=$(basename "$OUT"); T0=$(date +%s)
log() { echo "[$(($(date +%s)-T0))s] $*" >&2; }
DATA=$REPO/crates/shadowbox/tests/data
[ "$OPRUN" = none ] && OPRUN=
py() { INFER_ROWS=$ROWS python3 -I "$HERE/infer.py" "$1" "$OUT" "${@:2}"; }

export NV_SHARES=$OUT/shares.tsv
host_select "$HOSTS" local
for h in ${CAPS//,/ }; do [ -n "${HOST_CAP[${h%%:*}]:-}" ] && HOST_CAP[${h%%:*}]=${h#*:}; done
host_nvdisasm_check || exit 1

# Steps that re-ask known words on purpose get no answers from the store
# (pc1, where it exists: every base again, at a new address).
NV_LIVE_STEPS=" pc1 "
nv_step() { # nv_step NAME [DIR]: DIR/NAME.words.tsv -> DIR/NAME.nv.tsv: the
  # answer store, then the nvdisasm queue over the selected hosts for the rest
  local name=$1 d=${2:-$OUT} live= tag
  tag=${d#"$OUT"}; tag=${tag//\//-}
  case $NV_LIVE_STEPS in *" $name "*) live=1;; esac
  NVSTORE_LIVE=$live nv_store_take "$d/$name.words.tsv" "$d/$name.nv.tsv" || exit 1
  if [ -s "$d/$name.words.tsv.ask" ]; then
    NVADDR=1 host_queue_nv "$d/$name.words.tsv.ask" "$d/$name.nv.tsv" "$RUNID$tag-${name//\//-}" 128 \
      || { echo "nvdisasm failed in $name; partial results in $d" >&2; exit 1; }
    log "$name: $(wc -l < "$d/$name.nv.tsv") words asked"
  else : > "$d/$name.nv.tsv"; fi
  nv_store_put "$d/$name.words.tsv" "$d/$name.nv.tsv" "$d/$name" || exit 1
}

# One unit's layout passes on host H, to convergence (the loop below, for
# its rows only). A remote unit's files travel there once; each pass sends
# only the new table words back and their answers out.
unit_run() { local h=$1 u=$2 d=$OUT/units/$2 rd pass=1 t0=$SECONDS py
  if [ -z "${HOST_TARGET[$h]}" ]; then rd=$d py=$HERE/infer.py
  else rd=$(host_dir "$h" "$RUNID-units")/$u py=$(host_dir "$h" "$RUNID-units")/infer.py
    host_sh "$h" "rm -rf $rd && mkdir -p $rd" < /dev/null && host_sync_to "$h" "$rd/" -z "$d/bases.tsv" "$d/flips.nv.tsv" "$d/known.json" "$d/pc.rel.tsv" || return 1
  fi
  rm -f "$d/tables.words.tsv" "$d/tables.nv.tsv"
  while :; do
    host_run "$h" "python3 -I $py layout $rd" 2>> "$d/layout.log" || return 1
    if [ -n "${HOST_TARGET[$h]}" ]; then
      host_sync_back "$h" "$rd/tables.new.words.tsv" "$d/" -z && host_sync_back "$h" "$rd/tables.ask.words.tsv" "$d/" -z || return 1
    fi
    [ -s "$d/tables.new.words.tsv" ] || break
    if [ -s "$d/tables.ask.words.tsv" ]; then
      ( nv_step tables.ask "$d" ) 2>> "$d/layout.log" || return 1
      cat "$d/tables.ask.nv.tsv" >> "$d/tables.nv.tsv"
    fi
    cat "$d/tables.new.words.tsv" >> "$d/tables.words.tsv"; touch "$d/tables.nv.tsv"
    [ -z "${HOST_TARGET[$h]}" ] || host_sync_to "$h" "$rd/" -z "$d/tables.nv.tsv" "$d/tables.words.tsv" || return 1
    pass=$((pass + 1))
  done
  if [ -n "${HOST_TARGET[$h]}" ]; then
    local f; for f in layouts.tsv conflicts.tsv model.json; do host_sync_back "$h" "$rd/$f" "$d/" -z || return 1; done
    host_sh "$h" "rm -rf $rd" < /dev/null
  fi
  printf 'unit\t%s\t%s\t%s\t%s\t%s\t%s\n' "$u" "$h" "$(tr ',' '\n' < "$d/rows" | wc -l)" "$(wc -l < "$d/bases.tsv")" \
    "$pass" "$((SECONDS - t0))" >> "$NV_SHARES"
}
unit_feed() { # one worker slot on host H: take the next unit until none is left
  local h=$1 t u
  while :; do
    u=
    for t in "$OUT"/units/todo/*; do
      [ -e "$t" ] && mv "$t" "$OUT/units/run.$h.${t##*/}" 2>/dev/null && { u=${t##*/}; break; }
    done
    [ -n "$u" ] || return 0
    if unit_run "$h" "${u#*-}"; then rm -f "$OUT/units/run.$h.$u"
    else mv "$OUT/units/run.$h.$u" "$OUT/units/todo/$u"; touch "$OUT/units/failed.$h"
      echo "unit ${u#*-} failed on $h (see units/${u#*-}/layout.log); it goes back to the queue and this $h slot stops" >&2; return 1; fi
  done
}
layout_units() {
  local h k slots=0 round
  for h in "${HOSTS_SEL[@]}"; do slots=$((slots + HOST_CAP[$h])); done
  rm -rf "$OUT/units"
  python3 -I "$HERE/units.py" split "$OUT" "${INFER_UNITS:-$((slots * 2))}"
  for h in "${HOSTS_SEL[@]}"; do
    [ -n "${HOST_TARGET[$h]}" ] || continue
    host_sh "$h" "mkdir -p $(host_dir "$h" "$RUNID-units")" < /dev/null && host_sync_to "$h" "$(host_dir "$h" "$RUNID-units")/" "$HERE/infer.py" \
      && host_ensure "$h" python3 || touch "$OUT/units/failed.$h"
  done
  log "layout units on $(for h in "${HOSTS_SEL[@]}"; do printf '%s(%s) ' "$h" "${HOST_CAP[$h]}"; done)"
  for round in 1 2 3; do
    local -a pids=()
    for h in "${HOSTS_SEL[@]}"; do
      [ -e "$OUT/units/failed.$h" ] && continue
      for ((k = 0; k < HOST_CAP[$h]; k++)); do unit_feed "$h" & pids+=($!); done
    done
    [ ${#pids[@]} -gt 0 ] || break
    wait "${pids[@]}" || true
    compgen -G "$OUT/units/todo/*" > /dev/null || break
    rm -f "$OUT"/units/failed.*  # a slot that failed once gets one more try next round
  done
  if compgen -G "$OUT/units/todo/*" > /dev/null; then echo "layout units left after every host failed; see $OUT/units" >&2; exit 1; fi
  python3 -I "$HERE/units.py" merge "$OUT"
  rm -f "$OUT"/units/u*/flips.nv.tsv
  cat "$OUT"/units/u*/layout.log | awk '/^store recheck/ { t++; n += $4; m += $(NF-6) } /STORE MISMATCH/ { bad++ }
    END { printf "store rechecks in the layout units: %d takes, %d hits asked live, %d mismatches\n", t, n, m + bad }' >&2
  for h in "${HOSTS_SEL[@]}"; do [ -z "${HOST_TARGET[$h]}" ] || host_sh "$h" "rm -rf $(host_dir "$h" "$RUNID-units")" < /dev/null; done
  awk -F'\t' '$1 == "unit" { u[$3]++; r[$3] += $4; b[$3] += $5; s[$3] += $7 }
    END { for (h in u) printf "layout units: %s took %d units, %d rows, %d bases, %d busy seconds\n", h, u[h], r[h], b[h], s[h] }' "$NV_SHARES" >&2
}

if [ "$FROM" = start ]; then
log "samples: every manifest sample word of a root row (rows=$ROWS)"
awk -F'\t' -v sel="$ROWS" 'FNR==NR { if ($2 == "root" && $3 != "OOPS" && (sel == "all" || index("," sel ",", "," $1 ","))) r[$1] = 1; next }
  !/^#/ && ($1 in r) && !s[$4]++ { print $1 "\t" $1 "\t" $4 "\ts" }' \
  "$DATA/decoder-rows.tsv" "$DATA/decoder-samples.tsv" > "$OUT/samples.words.tsv"
nv_step samples
log "pick bases"; py pick "$DATA" "$OPRUN"
log "flips"; nv_step flips
log "pairs"; py pairs; nv_step pairs
fi
# branch targets are the word's address plus a value: ask every base again
# at a known address once; rows whose numbers moved get their flips and
# pairs asked again with addresses (pc.rel.tsv names them)
if [ ! -s "$OUT/pc1.nv.tsv" ]; then
  log "pc probe"; py pc1; nv_step pc1; py pc2
  if [ -s "$OUT/pc2.words.tsv" ]; then nv_step pc2; else : > "$OUT/pc2.nv.tsv"; fi
fi
# layout, then ask nvdisasm every value of each new selector field, until no
# field is left unenumerated (the tables only grow, so this converges).
# tables.nv.tsv is only nvdisasm text, kept as a cache across reruns.
SLOTS=0; for h in "${HOSTS_SEL[@]}"; do SLOTS=$((SLOTS + HOST_CAP[$h])); done
if [ "${INFER_UNITS:-}" != 0 ] && [ "$SLOTS" -gt 1 ]; then log "layout passes in units"; layout_units
else
rm -f "$OUT/tables.words.tsv"; PASS=1
while :; do
  log "layout pass $PASS"; py layout
  [ -s "$OUT/tables.new.words.tsv" ] || break
  [ -s "$OUT/tables.ask.words.tsv" ] && { nv_step tables.ask; cat "$OUT/tables.ask.nv.tsv" >> "$OUT/tables.nv.tsv"; }
  cat "$OUT/tables.new.words.tsv" >> "$OUT/tables.words.tsv"; PASS=$((PASS+1))
done
fi
log "compose"; py compose; nv_step validate
log "validate"; py validate
# the frozen evaluation set (frozen.py): the same held-out words every run
if nv_store_where; then
  if [ -n "${HOST_TARGET[$NVS_HOST]}" ]; then
    export FROZEN_SET=$OUT/.frozen-eval; rm -rf "$FROZEN_SET"; mkdir -p "$FROZEN_SET"
    host_sync_back "$NVS_HOST" "$NVS_DIR/frozen-eval/" "$FROZEN_SET/" -z 2> /dev/null || true
  else export FROZEN_SET=$NVS_DIR/frozen-eval; fi
fi
if [ -s "${FROZEN_SET:-/nonexistent}/words.tsv" ]; then
  log "validate on the frozen set"; python3 -I "$HERE/frozen.py" eval "$OUT"; nv_step frozen/validate
  python3 -I "$HERE/infer.py" validate "$OUT/frozen" 2> >(sed 's/^validate:/validate (frozen set):/' >&2)
else log "no frozen set yet (frozen.py make RUN draws it)"; fi
echo "RUNTIME $(($(date +%s)-T0))s  command: $0 --repo $REPO --oprun ${OPRUN:-none} --rows $ROWS --hosts $HOSTS --caps ${CAPS:-none} --out $OUT"
