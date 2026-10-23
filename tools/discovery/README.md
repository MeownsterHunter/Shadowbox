# Decoder discovery tools

These scripts reproduce the oracle sweeps and the bit-flip inference
described in [docs/decoder-methodology.md](../../docs/decoder-methodology.md).
They make instruction words, ask NVIDIA's disassembler (`nvdisasm`, the
oracle) what each word means, compare the answer with Shadowbox's decoder,
and work out each instruction family's bit layout from the answers.

The only inputs are Shadowbox's own decoder tables
(`crates/shadowbox/tests/data/decoder-rows.tsv` and `decoder-samples.tsv`)
and words the tools make up. No captured shaders or other outside data are
needed or used.

## Run it

```sh
tools/discovery/run.sh --work /path/to/work
```

This runs every step on this machine and ends with a summary: the sweep
buckets, the opcode-space counts, the validation rates and where each report
is. A full run is long: the row sweep is about 1.9 million words and
inference flips every bit of about 31,000 known words. To try it quickly,
limit it to a few rows:

```sh
tools/discovery/run.sh --work /path/to/work --rows 1997,1999,2006 \
  --depth 8 --samples 256 --no-opcode
```

Row numbers are line numbers of envytools `gm107.c`, the first column of
`decoder-rows.tsv`. Run `run.sh --help` for every option.

### Requirements

- Linux with bash, awk, `rg` (ripgrep) and Python 3 (standard library
  only; developed with Python 3.14).
- Rust and cargo, to build `sbdecode`, a small program over
  `shadowbox::decoder::decode_word`.
- `nvdisasm` from NVIDIA's CUDA toolkit (the runs in the methodology used
  CUDA 11.8). No GPU is needed. Install the toolkit, or only its
  `cuda_nvdisasm` component, from NVIDIA, then put `nvdisasm` on `PATH` or
  set `NVDISASM=/path/to/nvdisasm`. These tools never download, bundle or
  copy it.

### Resuming and rerunning

Each step writes into its own subdirectory of the work dir. When you run
`run.sh` again, a step whose outputs already exist, made with the same step
options, is skipped, and a step that runs makes every later step run too.
After you change the decoder or the tools, `--from STEP` reruns that step and
everything after it against the earlier outputs:

- `--from infer` redoes inference on the existing sweeps;
- `--from validate` redoes only the layout and the validation, reusing
  inference's flips.

`run.json` in the work dir records the source commit, the options, the
environment (paths relative to the work dir or the repo, other paths
reduced to a file name), the `nvdisasm` version, and each step's status and
time.

### Using more machines

By default everything runs locally, with one `nvdisasm` worker per CPU
(`DISCOVERY_WORKERS` changes that). To spread the `nvdisasm` batches over
other machines you can reach with key-based ssh, write a hosts file (see
[hosts.conf.example](hosts.conf.example)) and pass it:

```sh
tools/discovery/run.sh --work /path/to/work --hosts-file my-hosts.conf --hosts all
```

Each remote host needs `python3`, `rsync` and its own `nvdisasm`, the same
binary as the local one (the tools compare sha256 digests and refuse a host
whose copy differs). With `--hosts a,b,c` (or `all`) every host works at
once, as a work queue rather than an upfront split:

- every `nvdisasm` ask in inference is cut into chunks, and each host takes
  the next chunk whenever it finishes one;
- inference's layout passes run per group of rows ("units", heaviest
  first), each unit to its own convergence on whichever host has a free
  worker. Layout works row by row, so the merged `layouts.tsv`,
  `conflicts.tsv` and `model.json` are byte for byte what one process
  writes. `infer/shares.tsv` and the summary show what each host did.

A host that fails hands its chunk or unit back and the others finish it.
Only the words and the units' own inputs travel, and only `nvdisasm`'s text
and the units' layouts come back. Word generation, Shadowbox's decoding,
composing and validation run locally.
`tools/discovery/hosts.conf` is ignored by git, so a local hosts file there
stays out of commits.

### The answer store

`nvdisasm`'s answer for a word depends only on the `nvdisasm` binary, how it
is invoked, where the word sits and the word, so every answer is kept in one
insert-only sqlite store (`family-sweep/store.py`), keyed by the binary's
sha256, the invocation and batch format, and the word, with the address the
word was printed at. Every ask takes what the store already holds and sends
only the rest to `nvdisasm`; the per-run files come out exactly as before.
Each ask rechecks a random 1% of its hits (at least 200) against the live
`nvdisasm` and fails loudly, naming the key, on any mismatch.

The store is `$DISCOVERY_WORK/store` unless you say otherwise:
`--store DIR` (or `DISCOVERY_STORE`), `--store none` to turn it off, or a
`store=DIR` key on one host in the hosts file. With a store on another host,
only that host opens the sqlite file: words go to it over ssh, its answers
come back, and the recheck runs locally. Several runs, or several machines'
workers, can share one store at once.

### The frozen evaluation set

Validation composes its words from the run's own layout, so a layout change
redraws them and two runs' rates are measured on different words.
`family-sweep/frozen.py make RUN` draws a fixed held-out word list per row
once, with a recorded seed, into `<store>/frozen-eval`; every later
inference run is also scored on it (`infer/frozen/validate.tsv`, and a line
in the summary), so those numbers compare across runs.

## The steps

`run.sh` drives two scripts. Each can also be run alone; `--help` prints the
full description at the top of each script.

| Step | Command | Output (in the work dir) |
|------|---------|--------------------------|
| `sweep` | `family-sweep.sh` (row mode) | `sweep/report.txt`, `clusters.tsv`, `wrong_admissions.tsv`, `mismatch.tsv`, `attributes.tsv` |
| `opcode` | `family-sweep.sh --space opcode` | `opcode/report.txt`, `new_families.tsv`, `new_prefixes.tsv`, `unknown_known.tsv`, `row_gap.tsv`, `split.tsv` |
| `infer` | `family-infer.sh` | `infer/bases.tsv`, flips and pair flips (`*.words.tsv`, `*.nv.tsv`) |
| `validate` | `family-infer.sh --from layout` | `infer/layouts.tsv`, `conflicts.tsv`, `validate.tsv`, `validate-words.tsv`, `model.json` |

### Row sweep (`family-sweep.sh`)

Tests combinations inside the families the decoder already knows
(methodology, method 2).

1. `family-sweep/fields.py` reads the decoder tables and splits each row's
   bits into modifier, data and uncovered bits.
2. `family-sweep/anchors.py` picks, per row, operand values Shadowbox can
   decode (RZ, PT or attribute `0x80` instead of zero), using `sbdecode`.
3. `family-sweep/gen.py` makes the words: every modifier combination up to
   `--depth` bits, else `--samples` seeded random ones, plus uncovered-bit
   words, every attribute address and every system register.
4. A pilot of 64 words per row measures each row's reject rate, and
   `family-sweep/shard.py` splits the words across hosts by expected cost.
5. `family-sweep/nv.py` runs `nvdisasm -b SM52 -raw` over batches of 128
   words. `nvdisasm` stops at the first word it cannot decode, so `nv.py`
   marks that word rejected and reruns the rest.
6. `sbdecode` decodes every word with Shadowbox.
7. `family-sweep/analyze.py` sorts the differences into missing members,
   wrong admissions and name mismatches, clusters the missing ones, and
   writes `report.txt`.

### Opcode-space sweep (`family-sweep.sh --space opcode`)

Walks all 65,536 values of the top 16 bits instead of known rows, so it
finds families the tables do not list. `family-sweep/gen_opcode.py` makes
four words per prefix (all zeros, RZ/PT, and `--fills` random fills);
`family-sweep/analyze_opcode.py` reports new prefixes and families, Unknown
words of known rows, words no row matches, wrong admissions and prefixes
whose words disagree on the name. It is 262,144 words, and most are
rejected, each costing one `nvdisasm` process, so it is the slowest step per
word. `--no-opcode` skips it; inference then starts from the decoder
samples alone. `--prefixes FILE --seed S` resweeps only the listed prefixes
with new random fills, and `family-sweep/new_rows.py` turns the accepted
words no manifest row matches into rows of their own
(`decoder-rows-discovered.tsv` beside `decoder-rows.tsv`), which inference
then solves like any other row.

### Inference and validation (`family-infer.sh`)

Works out each row's bit layout from `nvdisasm`'s answers (methodology,
method 3). `family-sweep/infer.py` holds the text model and the rules.

1. `pick`: the known-valid words (givens) are the decoder samples
   `nvdisasm` accepts, plus the accepted words of the opcode-space run,
   one per row, name, modifier set and operand shape.
2. Every given with each of its 64 bits flipped goes to `nvdisasm`; each
   flip is classified (opcode, form, modifier, enum, flag, register,
   immediate, no effect).
3. `pairs`: neighbouring modifier bits are flipped together to find
   multi-bit fields.
4. `pc`: every given is asked again at a known address; a number that
   moved with the address is a branch target, so its row's flips are asked
   again with addresses and the number is learned relative to the word's
   own address.
5. `layout`: per row and operand form, bits become fields with roles and
   value tables; every value of every modifier field is asked of
   `nvdisasm` until no new field needs it. Disagreements go to
   `conflicts.tsv`.
6. `compose` and `validate`: new words are built from each layout
   (`INFER_PER_ROW`, default 16 per row), their text is predicted, and
   `nvdisasm` checks the prediction. `validate.tsv` gives per-row exact and
   canonical rates.
7. `frozen`: the same check on the frozen evaluation set, when the store
   has one.

`family-export.sh RUN OUT [ORACLE...]` turns a finished run into
`decoder-formats.tsv` (and the held samples beside it), the table
[tools/generate-decoder-formats](../generate-decoder-formats) reads.

## Environment

| Variable | Meaning | Default |
|----------|---------|---------|
| `NVDISASM` | the `nvdisasm` binary | `nvdisasm` on `PATH` |
| `DISCOVERY_WORK` | work dir when `--work` is not given | `<repo>/target/discovery` |
| `CARGO_TARGET_DIR` | where `sbdecode` builds | `$DISCOVERY_WORK/target` |
| `DISCOVERY_HOSTS` | hosts file when `--hosts-file` is not given | `tools/discovery/hosts.conf`, else local only |
| `DISCOVERY_WORKERS` | local `nvdisasm` workers | CPU count |
| `INFER_PER_ROW` | composed validation words per row | 16 |
| `INFER_UNITS` | layout units for the work queue (`0`: one local process) | twice the selected workers |
| `DISCOVERY_STORE` | the answer store: `DIR`, `HOST:DIR` or `none` (`--store`) | the hosts file's `store=` host, else `$DISCOVERY_WORK/store` |
| `DISCOVERY_RECHECK_WORKERS` | `nvdisasm` workers for the store's live recheck | 3 |
| `INFER_MEMORY_MAX` | optional memory cap for inference (for example `16G`), applied with `systemd-run --user --scope` | none |

## What these tools do not do

They never change the checkout, commit, install anything or use a GPU. They
do not ship or store `nvdisasm` or tables of its output in the repository;
the outputs stay in your work dir.
