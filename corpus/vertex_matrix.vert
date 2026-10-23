#version 450
layout(set = 0, binding = 0, std140) uniform Matrices { mat4 transform; } matrices;
layout(location = 0) in vec4 position;
void main() { gl_Position = matrices.transform * position; }
