#version 450
layout(location=0) in vec3 position; layout(location=1) in vec3 nrm; layout(location=2) in vec2 texcoord; layout(location=0) out vec3 world_position; layout(location=1) out vec3 world_normal; layout(location=2) out vec2 uv;
layout(set=0,binding=0,std140) uniform Camera { mat4 model; mat4 view; mat4 projection; } camera;
void main(){ vec4 world=camera.model*vec4(position,1); world_position=world.xyz; world_normal=mat3(camera.model)*nrm; uv=texcoord; gl_Position=camera.projection*camera.view*world; }
