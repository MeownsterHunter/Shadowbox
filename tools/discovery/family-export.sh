#!/usr/bin/env bash
# family-export.sh - export a finished family-infer.sh run as the decoder
# format table tools/generate-decoder-formats reads.
#
# THE IDEA
#   The inference knows each row's operand forms: fixed opcode bits, every
#   register, predicate, immediate, modifier and flag field, and how each value
#   prints. The decoder should know all of it, mechanically. One versioned
#   table (decoder-formats/3) carries it over, with no paths or hostnames.
#
# THE MODEL (infer.py export; the table's header documents every record)
#   variant  one base word of a row's operand form. Its fixed bits (opcode,
#            form, enumerated operands, immediates with no value model) are
#            the mask and value; the other fields are free, and any word
#            with the same fixed bits prints as the base's parsed text with
#            the free fields' values put in (infer.py predict + render,
#            exactly).
#   field    per form, one typed rule each: reg (file, slot, the constant
#            its value prints xor'd with, and the slots it prints again),
#            imm (slot, per-bit weights, constant added, sign), float
#            (slot, IEEE encoding), mod (value -> suffix names), flag,
#            none; fixed fields for reference (enum: value -> operand text).
#   rule     an imm's sign, negation or zeroing (printed 0x0), or a float's
#            encoding, keyed by selector or register values.
#   derive   a register atom printed as a reg field's value plus a constant,
#            or the zero register, as selector values say (a CAS pair).
#   token    a modifier no field owns, printed where selector values say.
#   refuse   the selector values, and whether immediates are zero, the
#            disassembler rejects; refuse-alone and refuse-rest: what key
#            values never seen fall back to.
#   render   per form: an operand shape decided by selector values.
#   zero, order, order-other, omit, flag  per row: how [RZ+0x0] prints,
#            modifier order counts (and those of the rows sharing the
#            mnemonic, for pairs the row never printed together), operands
#            left out at one value, flag bit shape turns.
#   Each variant carries its measured rate: of the held words it prints
#   (first variant in table order whose fixed bits match and whose prediction
#   exists), how many print exactly.
#   decoder-format-samples.tsv is every held word inside some variant's fixed
#   bits with each text the disassembler gave it (a word given two texts sat
#   at two addresses: a PC-relative target). The generator gates on it
#   (tools/generate-decoder-formats --samples).
#
# USAGE
#   family-export.sh RUN OUT [ORACLE...]
#     RUN     finished family-infer.sh run dir (model.json, bases.tsv, *.nv.tsv)
#     OUT     output dir
#     ORACLE  extra word<TAB>text files held as samples, e.g. the opcode-space
#             and row-sweep nv.tsv
#   About 4 to 9 minutes and 1.3 GB for 3.4 million held words. Runs in a
#   capped user scope where systemd-run exists (MemoryMax=16G, no swap,
#   nice 19). Reads RUN, writes only OUT.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)/family-sweep
[ $# -ge 2 ] || { sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 2; }
RUN=$1 OUT=$2; shift 2
mkdir -p "$OUT"
if command -v systemd-run > /dev/null; then
  exec systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 nice -n 19 \
    python3 -I "$HERE/infer.py" export "$RUN" "$OUT" "$@"
fi
exec nice -n 19 python3 -I "$HERE/infer.py" export "$RUN" "$OUT" "$@"
