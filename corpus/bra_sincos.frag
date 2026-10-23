#version 450
layout(location = 0) in float input_value;
layout(location = 0) out vec4 color;

void main() {
  float value = input_value;
  if (input_value > 0.0) {
    value += sin(input_value);
    value += cos(input_value);
    value += sin(value);
    value += cos(value);
    value += sin(value * 0.5);
    value += cos(value * 0.5);
  }
  color = vec4(value, sin(input_value), cos(input_value), 1.0);
}
