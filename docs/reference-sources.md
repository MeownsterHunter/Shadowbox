# Public reference sources

Shadowbox separates four questions:

1. Which bits select an instruction and its operands?
2. What observable behavior does the instruction have?
3. Can SPIR-V represent that behavior exactly?
4. Does the generated module produce the expected result on a Vulkan device?

No single reference answers all four.

## Instruction encoding

- [envytools](https://github.com/envytools/envytools) supplies public instruction
  tables and field definitions. The discovery pipeline pins
  [`envydis/gm107.c` at revision `f102b82381f3f11cee113d16374c87091db039d9`](https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c).
- [Mesa NAK](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/nouveau/compiler/nak)
  supplies an independently maintained open-source compiler implementation.
- NVIDIA's archived
  [CUDA Binary Utilities reference](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-binary-utilities/index.html)
  documents `nvdisasm` and the published instruction-set reference.

`nvdisasm` is used only to classify names and rendered fields. It is not a
behavioral oracle.

## SPIR-V and Vulkan

- [SPIR-V Registry](https://registry.khronos.org/SPIR-V/)
- [Vulkan SPIR-V environment](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)
- [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools)
- [Mesa NVK](https://docs.mesa3d.org/drivers/nvk.html)

Behavior is checked with project-owned CPU models, `spirv-val`, optimized-module
validation, and exact Vulkan observations. When the sources or experiments do
not establish one exact interpretation, the encoding remains unsupported.

## Project tooling

- [Decoder methodology](decoder-methodology.md)
- [Discovery pipeline](discovery-planner.md)
- [`shadowbox-discovery`](../tools/shadowbox-discovery)
- [Multi-host classifier tooling](../tools/discovery/README.md)
- [Project-authored corpus](test-corpus.md)
