#version 450
layout(location = 0) in float value;
layout(location = 0) out vec4 color;
void main() {
  int integer = int(value);
  color = vec4(float(integer));
}
