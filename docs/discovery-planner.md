# Discovery pipeline

Shadowbox maps the 64-bit instruction space through a source-locked funnel.

## 1. Build a constrained model

Public instruction tables define opcode regions, known field widths, enum
domains, and relationships between fields. The Sudoku stage solves those
constraints and emits canonical candidate assignments. Every artifact binds the
source revision and input digests.

## 2. Subtract the production decoder

Every candidate word is decoded by the current production decoder. Already
supported words are removed before external classification. This prevents
rediscovery and makes a stale source tree a hard error.

## 3. Enumerate targeted cubes

Candidates are clustered by opcode and semantic fields. Only unresolved bits
inside one coherent family are varied. A cube with `k` free bits contains
`2^k` words, so the planner caps `k`, total domain size, packet count, and output
size before enumeration.

Global word ownership is deterministic. If ranked cubes overlap, the first cube
owns the word and later cubes record the subtraction. This guarantees one
candidate word, one classifier answer, and one packet owner.

## 4. Classify and packetize

The external classifier processes the exact word list. Packetization requires
one non-conflicting answer per word and binds both the raw answer stream and its
normalized form. Invalid, unknown, or family-drifting answers are excluded or
quarantined instead of being coerced into a family.

Adjacent regions are bundled only when they share an exact semantic key. Bundle
limits keep qualification units bounded and prevent unrelated instruction
families from sharing evidence.

## 5. Qualify behavior

A bundle is implementation-ready only after permitted sources establish its
meaning. The implementation constructs typed state at the decoder boundary and
passes it through the IR to the emitter without downstream raw-bit
reclassification.

Qualification requires:

- exact admission and one-bit neighbor rejection;
- an independent CPU oracle, including exceptional and aliasing cases;
- validation of raw and optimized SPIR-V;
- exact Vulkan observations when the result is GPU-observable;
- regenerated decoder and rejection artifacts; and
- a clean, source-locked candidate revision.

If any semantic fact is missing, the bundle stops as a semantic decision rather
than entering the decoder.

## Reproducibility

The `shadowbox-discovery` command exposes the Sudoku catalog, hybrid plan,
packet, bundle, replay, preflight, and conveyor stages. Run its help output for
the exact command surface:

```text
cargo run -p shadowbox-discovery -- --help
```

Artifacts use canonical serialization and content digests. Replays re-acquire
the source lock, rebuild from the pinned inputs, and compare output bytes before
the next stage can proceed.
