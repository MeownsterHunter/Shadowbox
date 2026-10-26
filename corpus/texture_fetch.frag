#version 450
layout(location = 0) in vec2 coord;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    ivec2 size = textureSize(tex, 0);
    ivec2 p = ivec2(coord * vec2(size));
    color = texelFetch(tex, p, 0);
}
