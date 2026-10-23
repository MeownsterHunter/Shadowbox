#version 450
layout(location = 0) in vec2 coord;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    vec4 gathered = textureGather(tex, coord, 1);
    color = textureLodOffset(tex, coord, 1.0, ivec2(1, -1)) + gathered;
}
