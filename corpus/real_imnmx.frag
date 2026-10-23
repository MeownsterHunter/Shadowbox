#version 450
layout(location = 0) flat in ivec4 v;
layout(location = 0) out vec4 out_color;
void main() { out_color = vec4(min(v, ivec4(7))); }
