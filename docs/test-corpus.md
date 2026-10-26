# Project-authored test corpus

The checked-in corpus starts from GLSL fixtures authored for Shadowbox. Mesa's
NAK compiler produces the machine-code bytes, shader headers, SPIR-V inputs, and
text disassembly used by the tests.

Given a Mesa build directory containing `nak_produce`, regenerate with:

```text
bash tools/build-corpus.sh MESA_BUILD_DIRECTORY
bash tools/check-corpus.sh MESA_BUILD_DIRECTORY
```

Each GLSL fixture is compiled for Vulkan 1.2 and translated for SM 5.3.
`spirv-val --target-env vulkan1.2` validates every checked-in SPIR-V module. The
check script rebuilds the generated files in a temporary directory and compares
them byte for byte with the repository.

The corpus contains synthetic, project-authored fixtures only. It contains no
game shaders, proprietary captures, firmware, keys, or vendor binaries.
