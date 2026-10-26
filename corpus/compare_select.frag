#version 450
layout(location = 0) in float value;
layout(location = 0) out vec4 color;
void main() {
  float selected = value < 0.5 ? 1.0 : 0.0;
  color = vec4(selected);
}
