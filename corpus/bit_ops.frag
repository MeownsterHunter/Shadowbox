#version 450
layout(location = 0) flat in uint value;
layout(location = 0) out vec4 color;
void main() {
  uint mixed = (value << 2) ^ (value >> 1);
  color = vec4(float(mixed & 255u) / 255.0);
}
