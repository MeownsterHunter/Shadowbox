#version 450
layout(set = 0, binding = 0) uniform texture2D image;
layout(set = 0, binding = 1) uniform sampler image_sampler;
layout(location = 0) in vec2 coordinate;
layout(location = 0) out vec4 color;
void main() { color = texture(sampler2D(image, image_sampler), coordinate); }
