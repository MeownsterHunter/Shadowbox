#version 450
layout(location = 0) in vec4 left_value;
layout(location = 1) in vec4 right_value;
layout(location = 0) out vec4 color;
void main() {
  color = left_value + right_value;
}
