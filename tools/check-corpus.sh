cp "$root"/corpus/*.{frag,vert} "$tmp"/
for source in "$tmp"/*.{frag,vert}; do
  [[ -e "$source" ]] || continue
  name=${source##*/}; stage=${name##*.}; name=${name%.*}
  glslc --target-env=vulkan1.2 -fshader-stage="$stage" "$source" -o "$tmp/$name.spv"
  "$1/src/nouveau/compiler/nak_produce" "$tmp/$name.spv" "$tmp/$name.bin" "$tmp/$name.hdr" "$tmp/$name.asm" "$stage" 53
