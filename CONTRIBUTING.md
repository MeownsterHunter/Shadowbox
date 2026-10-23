# Contributing

Thanks for looking. This project has one unusual requirement, so please read this first.

## Before your first change

Read [CLEAN-ROOM.md](CLEAN-ROOM.md). By contributing you state that:

- you wrote the change yourself, or it comes from a source the clean-room rules allow and you say which,
- you did not use any proprietary development kit, any material under a non-disclosure agreement, any leaked material, or any proprietary source or decompiled code, from any party,
- you have not had access to such material, or your change does not touch what it covers,
- you agree to the contributor agreement in [CLA.md](CLA.md), which lets the project include your work under its two public licenses and under commercial licenses.

Add a `Signed-off-by` line to your commits (`git commit -s`) to record that statement, and say in your first pull request that you agree to the contributor agreement.

## What a change needs

- A provenance note for any new or changed API behaviour. See [docs/provenance/README.md](docs/provenance/README.md).
- A test that does not depend on any game or proprietary file.
- `cargo fmt`, `cargo clippy -- -D warnings` and `cargo test` passing.

## What will be declined

- Anything containing or derived from proprietary material.
- Game-specific hacks, game files, keys, firmware, or instructions for obtaining them.
- Changes whose origin cannot be explained.

## Reporting a concern

If you think something in the repository should not be here, open an issue or contact a maintainer privately. It will be looked at promptly and handled as described in the clean-room rules.
