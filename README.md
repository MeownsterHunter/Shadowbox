# Shadowbox

Shadowbox is a Rust translator for precompiled NVIDIA Maxwell and Pascal shader
machine code. It decodes supported instructions into a typed intermediate
representation and emits Vulkan-compatible SPIR-V for fragment, vertex,
compute, geometry, and tessellation stages.

The decoder is intentionally conservative. An encoding remains unsupported
until its bit boundary and behavior are established by public sources and
independent tests. Unsupported instructions fail closed instead of being
approximated.

## Workspace

- `shadowbox` contains the decoder, typed IR, control-flow analysis, SPIR-V
  emitter, optimizer, cache, and validation tests.
- `shadowbox-cli` provides translation, disassembly, coverage, cache, scanning,
  gate, and value-comparison commands.
- `shadowbox-novena` and `shadowbox-novena-host` expose the translator through
  Novena's Rust and C interfaces.
- `shadowbox-discovery` contains the source-locked Sudoku, targeted-cube,
  packetization, and conveyor tooling used to study the instruction format.

The Novena library needed by the adapter crates is included under
`vendor/novena` so the public workspace does not depend on a machine-local
checkout.

## Build and test

Install a Rust toolchain and SPIRV-Tools, including `spirv-val`, then run:

```text
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

GPU tests are opt-in because they require a Vulkan device and driver:

```text
cargo test -p shadowbox --features gpu-check --test gpu_exact -- --ignored
```

Build the command-line tool with:

```text
cargo build --release -p shadowbox-cli
```

The resulting binary supports `translate`, `disasm`, `coverage`, `gate`,
`value-diff`, cache management, and ahead-of-time shader discovery. Run
`shadowbox help` for the current command surface.

## Methodology

The mapping process is documented in:

- [Decoder methodology](docs/decoder-methodology.md)
- [Discovery pipeline](docs/discovery-planner.md)
- [Public reference sources](docs/reference-sources.md)
- [Project-authored test corpus](docs/test-corpus.md)
- [Current decoder coverage](docs/decoder-coverage.md)

The repository contains project-authored fixtures only. It does not contain
game shaders, proprietary captures, firmware, keys, vendor binaries, or private
machine artifacts.

## Clean-room policy

Read [CLEAN-ROOM.md](CLEAN-ROOM.md) before contributing. Required third-party
copyright and license notices remain beside adapted source and in `NOTICE` and
`LICENSES`.

## License

Shadowbox is available under either the PolyForm Noncommercial License 1.0.0
or the GNU Affero General Public License, version 3 or later. See
[LICENSE.md](LICENSE.md) and [COMMERCIAL.md](COMMERCIAL.md).
