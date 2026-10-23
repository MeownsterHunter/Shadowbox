# Clean-room policy

Shadowbox is implemented from public, permitted sources and project-owned
experiments. Do not use proprietary SDKs, headers, documentation, compilers,
leaked material, game data, firmware, keys, vendor binaries, or decompiled
proprietary code.

GPL-licensed emulator projects and their forks are excluded as implementation
and research sources. Public specifications and permissively licensed source
code may be used only within their license terms. Adapted code must retain the
applicable copyright and permission notice.

## Permitted sources

- [envytools](https://github.com/envytools/envytools), under the MIT license,
  for public instruction tables and field definitions.
- [Mesa](https://gitlab.freedesktop.org/mesa/mesa), under the MIT license, for
  Nouveau, NAK, and NVK encoding and compiler behavior.
- The public Vulkan and SPIR-V specifications.
- The public maxas wiki and NVIDIA CUDA Binary Utilities documentation.
- Project-authored experiments using synthetic inputs.
- Permissively licensed archive readers, only for archive and container layout.

`nvdisasm` may be used as a black-box classifier on project-authored instruction
words. Its output can establish a spelling or rendered field, but never
instruction behavior.

## Required evidence

Every change that adds instruction behavior must identify the public source and
revision, define the exact admitted bit boundary, and test rejected neighboring
states. Behavioral evidence must include an independent CPU oracle, raw and
optimized SPIR-V validation, and a Vulkan result when the operation can be
observed on a GPU.

Repository documentation uses repository-relative paths and public source URLs.
Do not commit checkout paths, machine names, usernames, task identifiers,
worktree names, shell transcripts, or temporary artifact locations.

## Corpus rules

Corpus inputs must be authored for this project. Do not commit game shaders,
proprietary captures, firmware, keys, or vendor binaries. Generated fixtures
must have a documented public producer and a reproducible command.

When the origin, license, boundary, or semantics of a fact is uncertain, leave
the encoding unsupported.
