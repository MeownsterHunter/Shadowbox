# Wide-row decoder validation

This file summarizes the checked-in [result ledger](../crates/shadowbox/tests/data/decoder-wide-results.tsv). The raw `nvdisasm` answers remain local and are bound by SHA-256 in that ledger.

These are sampled covering-set measurements, not exhaustive closure. A `T(k,n)` row means zero errors among exactly `n` generated words at interaction strength `k` and the recorded payload strata.

- Rows measured: 247 (215 qualified, 4 format gaps, 28 blocked by hidden interactions, 0 decoder errors).
- Words checked: 963990338.
- Format-table gaps/errors: 16928755 missing, 0 wrong admissions, 755 misprints.
- Shipped-decoder wrong admissions: 0.

A blocked row is evidence that the proposed modifier boundary is incomplete. It is not decoder support and is not silently promoted.
