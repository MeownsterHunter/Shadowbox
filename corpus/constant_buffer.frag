#version 450
layout(set = 0, binding = 0, std140) uniform Params { vec4 tint; };
layout(location = 0) out vec4 color;
void main() { color = tint; }
