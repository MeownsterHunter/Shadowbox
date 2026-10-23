#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
producer=${1:?usage: $0 MESA_BUILD_DIR}/src/nouveau/compiler/nak_produce
for source in "$root"/corpus/*.{frag,vert}; do
  [[ -e "$source" ]] || continue
  name=${source##*/}; stage=${name##*.}; name=${name%.*}
  glslc --target-env=vulkan1.2 -fshader-stage="$stage" "$source" -o "$root/corpus/$name.spv"
  "$producer" "$root/corpus/$name.spv" "$root/corpus/$name.bin" "$root/corpus/$name.hdr" "$root/corpus/$name.asm" "$stage" 53
done
