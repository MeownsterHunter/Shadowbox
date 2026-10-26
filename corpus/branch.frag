#version 450
layout(location = 0) in float selector;
layout(location = 0) out vec4 color;
void main() {
  if (selector > 0.0)
    color = vec4(1.0, 0.0, 0.0, 1.0);
  else
    color = vec4(0.0, 0.0, 1.0, 1.0);
}
