# Corpus notes

Uniform-buffer reads are emitted as constant-buffer loads. Their assembly must
use `c[N][offset]` operands or `LDC`; `ld.global` is not an acceptable
representation for a UBO fixture.

## Counts

The repository tracks 33 `.bin` inputs in this directory. `translates.txt` names the 24 inputs
for which the translation gate currently promises deterministic, Vulkan-valid
SPIR-V. The remaining nine are retained as decoder evidence and are not counted
as translation successes. Older text reported 27 translated fixtures without a
matching manifest; the manifest-backed count is authoritative.
