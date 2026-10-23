# hosts.sh - the host table and helpers for the discovery tools. Sourced, not
# run:  . tools/discovery/hosts.sh
#
# THE IDEA
#   nvdisasm batches are embarrassingly parallel, and rejected words make
#   them slow (one nvdisasm process per rejected word). By default every
#   batch runs on this machine. An optional hosts file spreads the batches
#   over other machines you can reach with plain key-based ssh: only the
#   words travel out and only nvdisasm text comes back. Word generation,
#   Shadowbox classification and analysis always run locally.
#   A tool takes a host selector (its own flag or env var) and calls these
#   helpers instead of hardcoding a machine.
#
# THE HOSTS FILE  ($DISCOVERY_HOSTS, default tools/discovery/hosts.conf;
#   when it does not exist, the only host is 'local')
#   One host per line, '#' starts a comment:
#     NAME WORKERS [speed=S] [ssh=TARGET] [dir=DIR] [nvdisasm=PATH] [store=DIR]
#   NAME      the label --hosts uses. 'local' is this machine, without ssh.
#   WORKERS   parallel nvdisasm processes on that host.
#   speed     per-worker nvdisasm throughput relative to local (default
#             1.0); shards are balanced on workers x speed.
#   ssh       ssh destination (default NAME), e.g. user@host.
#   dir       work dir on that host (default discovery-work, relative to
#             the remote home).
#   nvdisasm  nvdisasm on that host (default: nvdisasm on its PATH).
#   store     this host keeps the shared answer store in DIR (see ANSWER
#             STORE below); at most one host has it.
#   Without a 'local' line, local still exists with $DISCOVERY_WORKERS
#   workers (default: the CPU count). Remote hosts need python3 and their
#   own nvdisasm; nothing is installed or copied for you.
#   See hosts.conf.example.
#
# SELECTORS (host_select SEL DEFAULT)
#   local | NAME | a,b,c   exactly those hosts, at their table workers.
#   auto   the one host with the most free capacity (probe: nproc minus the
#          1-minute loadavg, over ssh, capped at its workers); ties go to
#          DEFAULT, then table order.
#   all    every host with at least one free worker, each at its free
#          capacity; if none has one, DEFAULT with 1 worker.
#   Result: HOSTS_SEL (array, in table order) and HOST_CAP[host] (workers).
#
# HELPERS
#   host_sh H CMD            plain shell on H (stdin passes through)
#   host_run H CMD           heavy work on H, at nice 19
#   host_dir H TOOL          the tool's work dir on H, as H's shell sees it
#   host_path H P            rsync spelling of P on H (target:P)
#   host_sync_to H DEST SRC.. [rsync opts first]  rsync -a sources to H:DEST
#   host_sync_back H SRC DEST [rsync opts after]  rsync -a H:SRC to local DEST
#   host_free H              free workers now (0 when unreachable)
#   host_ensure H ITEM..     nvdisasm or a command (python3 rsync ...) checked
#                            where heavy work runs; a missing one is
#                            reported, never installed.
#   host_split IN PREFIX     deal IN's lines to PREFIX.<host>, weighted by
#                            HOST_CAP x speed, interleaved so costly runs of
#                            neighbouring lines spread.
#   host_parallel FN         run `FN host` for every selected host at once;
#                            fails if any fails. Merge results with cat.
#   host_nv H IN OUT NAME [CH]  nv.py (family-sweep/) over IN on H (NVADDR=1
#                            passed through: texts carry their address) with
#                            HOST_CAP workers in <dir>/NAME; OUT gets the
#                            nvdisasm text. Only the given words travel.
#   host_queue_nv IN OUT NAME [CH]  the same over every selected host as a
#                            work queue: IN is cut into chunks, and each host
#                            takes the next chunk whenever it finishes one
#                            (work stealing, no upfront split). A host that
#                            fails gives its chunk back and stops; the others
#                            finish the queue. $NV_SHARES, when set, gets one
#                            line per host: nv NAME host chunks words seconds.
#   host_nvdisasm_check      every selected host's nvdisasm has the same
#                            sha256 as this machine's (the store key's).
# Never sudo, never kill anything; ssh runs with BatchMode (keys only).
HOSTS_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOSTS_FILE=${DISCOVERY_HOSTS:-$HOSTS_TOOLS/hosts.conf}
HOSTS_ALL=()
declare -A HOST_TARGET=() HOST_DISCOVERY=() HOST_WORKERS=() HOST_SPEED=() HOST_ND=()
STORE_HOST= STORE_DIR=
declare -A HOST_CAP=()
HOSTS_SEL=()

hosts_load() { local name workers kv k v
  if [ -f "$HOSTS_FILE" ]; then
    while read -r name workers kv; do
      case $name in ''|'#'*) continue;; esac
      [[ $workers =~ ^[0-9]+$ ]] || { echo "$HOSTS_FILE: '$name' needs a worker count" >&2; return 2; }
      HOSTS_ALL+=("$name"); HOST_WORKERS[$name]=$workers; HOST_SPEED[$name]=1.0
      if [ "$name" = local ]; then HOST_TARGET[$name]=; HOST_DISCOVERY[$name]=
      else HOST_TARGET[$name]=$name; HOST_DISCOVERY[$name]=discovery-work; HOST_ND[$name]=nvdisasm; fi
      for kv in ${kv%%#*}; do k=${kv%%=*} v=${kv#*=}
        case $k in
          speed) HOST_SPEED[$name]=$v;; ssh) HOST_TARGET[$name]=$v;;
          dir) HOST_DISCOVERY[$name]=$v;; nvdisasm) HOST_ND[$name]=$v;;
          store) STORE_HOST=$name STORE_DIR=$v;;
          *) echo "$HOSTS_FILE: unknown key '$k' for '$name'" >&2; return 2;;
        esac
      done
    done < "$HOSTS_FILE"
  fi
  if [ -z "${HOST_WORKERS[local]:-}" ]; then
    HOSTS_ALL=(local "${HOSTS_ALL[@]}"); HOST_TARGET[local]=; HOST_DISCOVERY[local]=
    HOST_WORKERS[local]=${DISCOVERY_WORKERS:-$(nproc)}; HOST_SPEED[local]=1.0
  fi
}
hosts_load || return 2

host_known() { [ -n "${HOST_WORKERS[$1]:-}" ] || { echo "unknown host '$1' (${HOSTS_ALL[*]})" >&2; return 2; }; }
# One shared ssh connection per host (ControlMaster) keeps the many short
# calls of a queue run cheap.
host_ssh_opts() { echo "-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ControlMaster=auto -o ControlPersist=60 -o ControlPath=${XDG_RUNTIME_DIR:-/tmp}/discovery-ssh-%C"; }
host_sh() { local h=$1; shift
  if [ -z "${HOST_TARGET[$h]}" ]; then bash -c "$1"
  else ssh $(host_ssh_opts "$h") "${HOST_TARGET[$h]}" "$1"; fi
}
host_run() { local h=$1; shift
  if [ -z "${HOST_TARGET[$h]}" ]; then nice -n 19 bash -c "$1" < /dev/null
  else ssh -n $(host_ssh_opts "$h") "${HOST_TARGET[$h]}" "nice -n 19 bash -c $(printf %q "$1")"; fi
}
host_dir() { echo "${HOST_DISCOVERY[$1]}/$2"; }
host_path() { local p=$2
  if [ -z "${HOST_TARGET[$1]}" ]; then echo "$p"; else p=${p#\~/}; echo "${HOST_TARGET[$1]}:$p"; fi
}
host_sync_to() { local h=$1 dest=$2; shift 2
  rsync -a -e "ssh $(host_ssh_opts "$h")" "$@" "$(host_path "$h" "$dest")"
}
host_sync_back() { local h=$1 src=$2 dest=$3; shift 3
  rsync -a -e "ssh $(host_ssh_opts "$h")" "$@" "$(host_path "$h" "$src")" "$dest"
}
host_free() { local h=$1 n
  n=$(host_sh "$h" 'echo $(nproc) $(cut -d" " -f1 /proc/loadavg)' < /dev/null 2>/dev/null) || { echo 0; return; }
  awk -v w="${HOST_WORKERS[$h]}" '{f=int($1-$2); if(f<0)f=0; if(f>w)f=w; print f}' <<< "$n"
}
host_select() { local sel=$1 def=$2 h best= bestc=-1 c
  HOSTS_SEL=(); HOST_CAP=()
  case $sel in
    auto|all)
      declare -A free=()
      for h in "${HOSTS_ALL[@]}"; do free[$h]=$(host_free "$h"); done
      if [ "$sel" = auto ]; then
        for h in "$def" "${HOSTS_ALL[@]}"; do c=${free[$h]}; [ "$c" -gt "$bestc" ] && { best=$h; bestc=$c; }; done
        HOSTS_SEL=("$best"); HOST_CAP[$best]=$(( bestc > 0 ? bestc : 1 ))
      else
        for h in "${HOSTS_ALL[@]}"; do [ "${free[$h]}" -ge 1 ] && { HOSTS_SEL+=("$h"); HOST_CAP[$h]=${free[$h]}; }; done
        [ ${#HOSTS_SEL[@]} -gt 0 ] || { HOSTS_SEL=("$def"); HOST_CAP[$def]=1; }
      fi;;
    *)
      local IFS=,; for h in $sel; do host_known "$h" || return 2; done
      for h in "${HOSTS_ALL[@]}"; do case ",$sel," in *",$h,"*) HOSTS_SEL+=("$h"); HOST_CAP[$h]=${HOST_WORKERS[$h]};; esac; done;;
  esac
  local s=; for h in "${HOSTS_SEL[@]}"; do s="$s $h(${HOST_CAP[$h]})"; done
  echo "hosts ($sel):$s" >&2
}
# nvdisasm: locally $NVDISASM, else nvdisasm on PATH; remotely the hosts file's
# nvdisasm= (default nvdisasm on that host's PATH). Install it from NVIDIA's
# CUDA toolkit; these tools never download or copy it.
host_nvdisasm() { if [ -z "${HOST_TARGET[$1]}" ]; then command -v "${NVDISASM:-nvdisasm}" || echo "${NVDISASM:-nvdisasm}"; else echo "${HOST_ND[$1]}"; fi; }
host_ensure() { local h=$1 i cmds=; shift
  for i in "$@"; do case $i in
    nvdisasm)
      if [ -z "${HOST_TARGET[$h]}" ]; then
        [ -x "$(host_nvdisasm "$h")" ] || { echo "$h: nvdisasm not found: install NVIDIA's CUDA toolkit and put nvdisasm on PATH or set NVDISASM" >&2; return 1; }
      else
        host_sh "$h" "command -v $(host_nvdisasm "$h") > /dev/null" < /dev/null \
          || { echo "$h: nvdisasm not found there: install it, or set nvdisasm= in $HOSTS_FILE" >&2; return 1; }
      fi;;
    *) cmds="$cmds $i";;
  esac; done
  [ -n "$cmds" ] || return 0
  local miss; miss=$(host_run "$h" "for c in $cmds; do command -v \$c > /dev/null || echo \$c; done")
  [ -z "$miss" ] || { echo "$h: missing prerequisite(s): $(echo $miss) (install by hand; this never installs anything)" >&2; return 1; }
}
host_split() { local in=$1 prefix=$2 h spec=
  for h in "${HOSTS_SEL[@]}"; do spec="$spec $h:${HOST_CAP[$h]}:${HOST_SPEED[$h]}"; : > "$prefix.$h"; done
  awk -v spec="$spec" -v p="$prefix" '
    BEGIN { n = split(spec, s, " "); for (i = 1; i <= n; i++) { split(s[i], f, ":"); h[i] = f[1]; w[i] = f[2] * f[3]; l[i] = 0 } }
    { b = 1; for (i = 2; i <= n; i++) if ((l[i] + 1) / w[i] < (l[b] + 1) / w[b]) b = i; l[b]++; print > (p "." h[b]) }' "$in"
}
host_parallel() { local fn=$1 h p fail=0; local -a pids=()
  for h in "${HOSTS_SEL[@]}"; do "$fn" "$h" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  return $fail
}
host_nv() { local h=$1 in=$2 out=$3 name=$4 ch=${5:-128}
  local np=${HOST_CAP[$h]:-${HOST_WORKERS[$h]}} nvpy=$HOSTS_TOOLS/family-sweep/nv.py
  host_ensure "$h" nvdisasm || return 1
  if [ -z "${HOST_TARGET[$h]}" ]; then
    local t; t=$(dirname "$out")/nvtmp.$h.$$
    NVADDR=${NVADDR:-} ND=$(host_nvdisasm "$h") NVTMP=$t NP=$np CH=$ch nice -n 19 python3 -I "$nvpy" "$in" "$out" < /dev/null
    local rc=$?; rm -rf "$t"; return $rc
  fi
  local d=${HOST_DISCOVERY[$h]}/$name
  host_sh "$h" "mkdir -p $d && rm -f $d/out.tsv" < /dev/null || return 1
  host_sync_to "$h" "$d/nv.py" "$nvpy" && host_sync_to "$h" "$d/in.tsv" "$in" || return 1
  host_run "$h" "cd $d && rm -rf nvtmp && NVADDR=${NVADDR:-} ND=$(host_nvdisasm "$h") NVTMP=\$PWD/nvtmp NP=$np CH=$ch python3 -I nv.py in.tsv out.tsv && rm -rf nvtmp" || return 1
  host_sync_back "$h" "$d/out.tsv" "$out"
}
host_queue_nv() { local in=$1 out=$2 name=$3 ch=${4:-128} n slots=0 per h round
  local q; q=$(dirname "$out")/.queue.$(basename "$out").$BASHPID
  rm -rf "$q"; mkdir -p "$q/todo" "$q/done"
  n=$(wc -l < "$in")
  if [ "$n" -eq 0 ]; then : > "$out"; rm -rf "$q"; return 0; fi
  for h in "${HOSTS_SEL[@]}"; do slots=$((slots + HOST_CAP[$h])); done
  per=$(( n / (slots > 0 ? slots * 4 : 1) )); [ $per -lt $((ch * 4)) ] && per=$((ch * 4)); [ $per -gt 16384 ] && per=16384
  split -l "$per" -d -a 6 "$in" "$q/todo/c"
  _queue_feed() { local h=$1 c k=0 w=0 t0=$SECONDS id
    for c in "$q"/todo/c*; do
      id=${c##*/}; mv "$c" "$q/run.$h.$id" 2>/dev/null || continue
      if host_nv "$h" "$q/run.$h.$id" "$q/done/$id" "$name-$id" "$ch"; then
        k=$((k + 1)); w=$((w + $(wc -l < "$q/run.$h.$id"))); rm -f "$q/run.$h.$id"
        [ -n "${HOST_TARGET[$h]}" ] && host_sh "$h" "rm -rf ${HOST_DISCOVERY[$h]}/$name-$id" < /dev/null
      else
        mv "$q/run.$h.$id" "$c"; rm -f "$q/done/$id"; touch "$q/failed.$h"
        echo "$h: nvdisasm chunk failed; it goes back to the queue and $h stops taking chunks" >&2; break
      fi
    done
    [ -n "${NV_SHARES:-}" ] && [ $k -gt 0 ] && printf 'nv\t%s\t%s\t%s\t%s\t%s\n' "$name" "$h" "$k" "$w" "$((SECONDS - t0))" >> "$NV_SHARES"
    return 0
  }
  for round in 1 2 3; do
    local -a pids=()
    for h in "${HOSTS_SEL[@]}"; do [ -e "$q/failed.$h" ] || { _queue_feed "$h" & pids+=($!); }; done
    [ ${#pids[@]} -gt 0 ] || break
    wait "${pids[@]}"
    compgen -G "$q/todo/c*" > /dev/null || break
  done
  if compgen -G "$q/todo/c*" > /dev/null; then echo "nvdisasm queue $name: chunks left after every host failed; see $q" >&2; return 1; fi
  cat "$q"/done/c* > "$out"; rm -rf "$q"
}
host_nvdisasm_check() { local h want got
  want=$(sha256sum "$(host_nvdisasm local)" | cut -d' ' -f1)
  for h in "${HOSTS_SEL[@]}"; do
    [ -n "${HOST_TARGET[$h]}" ] || continue
    got=$(host_sh "$h" "sha256sum $(host_nvdisasm "$h")" < /dev/null | cut -d' ' -f1)
    [ "$got" = "$want" ] || { echo "$h: nvdisasm sha256 ${got:-unreadable} differs from this machine's $want; its answers would be stored under the wrong key. Give every host the same nvdisasm." >&2; return 1; }
  done
}
# ANSWER STORE (family-sweep/store.py has the format, keying and recheck)
#   Where: $DISCOVERY_STORE as DIR (here), HOST:DIR (a hosts-file host) or
#   none (no store: every word is asked, nothing kept); unset, the host with
#   store= in the hosts file, else $DISCOVERY_WORK/store here. Only the store
#   host opens the sqlite: a remote store gets the words over ssh and sends
#   the answers back (store.py lookup/insert run there); the sample recheck
#   always runs here with this machine's nvdisasm.
#   nv_store_take IN OUT     OUT.hit gets the store's answers to IN's words
#                            (a sample rechecked live; a mismatch fails),
#                            IN.ask the lines whose word is still to ask.
#   nv_store_put IN OUT SRC  OUT holds nv.py's answers to IN.ask, asked with
#                            NVADDR=1: they are stored, and OUT becomes every
#                            answer to IN in the caller's format (as before).
nv_store_where() { local s=${DISCOVERY_STORE:-}
  NVS_HOST=local NVS_DIR=
  if [ -z "$s" ] && [ -n "$STORE_HOST" ]; then NVS_HOST=$STORE_HOST NVS_DIR=$STORE_DIR
  elif [ -z "$s" ]; then NVS_DIR=${DISCOVERY_WORK:-$HOSTS_TOOLS/../../target/discovery}/store
  elif [ "$s" = none ]; then return 1
  elif [[ $s == *:* ]] && [ -n "${HOST_WORKERS[${s%%:*}]:-}" ]; then NVS_HOST=${s%%:*} NVS_DIR=${s#*:}
  else NVS_DIR=$s; fi
}
nv_store_tmp() { echo "${DISCOVERY_WORK:-$HOSTS_TOOLS/../../target/discovery}/tmp"; }
nv_store_py() { local nd; nd=$(host_nvdisasm local); mkdir -p "$(nv_store_tmp)"
  ND=$nd NVTMP=$(nv_store_tmp)/recheck.$BASHPID.$RANDOM NP=${DISCOVERY_RECHECK_WORKERS:-3} NVADDR=${NVADDR:-} \
    NVSTORE_LIVE=${NVSTORE_LIVE:-} DISCOVERY_STORE=${NVS_DIR:-} nice -n 19 python3 -I "$HOSTS_TOOLS/family-sweep/store.py" "$@"
}
nv_store_remote() { # nv_store_remote CMD ARGS..: store.py on the store host, in the dir $T
  local digest; digest=$(sha256sum "$(host_nvdisasm local)" | cut -d' ' -f1)
  host_run "$NVS_HOST" "cd $T && DISCOVERY_STORE=$NVS_DIR NVSTORE_VERSION=$digest NVADDR=${NVADDR:-} NVSTORE_LIVE=${NVSTORE_LIVE:-} NVSTORE_LABEL=$LABEL python3 -I store.py $*"
}
nv_store_take() { local in=$1 out=$2 T LABEL; LABEL=$(basename "$in")
  if ! nv_store_where; then cp "$in" "$in.ask"; : > "$out.hit"; return 0; fi
  if [ -z "${HOST_TARGET[$NVS_HOST]}" ]; then nv_store_py take "$in" "$out"; return; fi
  T=$NVS_DIR/tmp/${HOSTNAME:-here}-$BASHPID-$RANDOM
  host_sh "$NVS_HOST" "mkdir -p $T" < /dev/null && host_sync_to "$NVS_HOST" "$T/" -z "$HOSTS_TOOLS/family-sweep/store.py" \
    && host_sync_to "$NVS_HOST" "$T/words.tsv" -z "$in" && nv_store_remote lookup words.tsv out \
    && mkdir -p "$out.got" && host_sync_back "$NVS_HOST" "$T/" "$out.got/" -z --exclude=store.py --exclude=words.tsv || return 1
  mv "$out.got/out.hit" "$out.hit"; mv "$out.got/out.check" "$out.check"; mv "$out.got/words.tsv.ask" "$in.ask"; rm -rf "$out.got"
  host_sh "$NVS_HOST" "rm -rf $T" < /dev/null
  nv_store_py recheck "$out" "$(basename "$in")"
}
nv_store_put() { local in=$1 out=$2 src=$3 T LABEL; LABEL=$(basename "$out")
  if ! nv_store_where; then NVS_DIR= nv_store_py assemble "$in" "$out"; return; fi
  if [ -z "${HOST_TARGET[$NVS_HOST]}" ]; then nv_store_py put "$in" "$out" "$src"; return; fi
  T=$NVS_DIR/tmp/${HOSTNAME:-here}-$BASHPID-$RANDOM
  if [ -s "$out" ]; then
    host_sh "$NVS_HOST" "mkdir -p $T" < /dev/null && host_sync_to "$NVS_HOST" "$T/" -z "$HOSTS_TOOLS/family-sweep/store.py" \
      && host_sync_to "$NVS_HOST" "$T/new.tsv" -z "$out" && nv_store_remote insert new.tsv "$(printf %q "$src")" || return 1
    host_sh "$NVS_HOST" "rm -rf $T" < /dev/null
  fi
  nv_store_py assemble "$in" "$out"
}
