#version 450
layout(set = 0, binding = 0) uniform textureCube image;
layout(set = 0, binding = 1) uniform sampler image_sampler;
layout(location = 0) in vec3 coordinate;
layout(location = 0) out vec4 color;
void main() { color = texture(samplerCube(image, image_sampler), coordinate); }
