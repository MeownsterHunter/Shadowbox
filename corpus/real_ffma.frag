#version 450
layout(location = 0) in vec4 v;
layout(location = 0) out vec4 out_color;
void main() { out_color = fma(v, vec4(1.25), vec4(0.5)); }
