#version 450
layout(location=0) in vec3 position; layout(location=1) in vec4 weights; layout(location=2) in uvec4 bones_in; layout(location=0) out vec3 world_position;
layout(set=0,binding=0) uniform Bones { mat4 bones[64]; } skeleton;
void main(){ mat4 skin=weights.x*skeleton.bones[bones_in.x]+weights.y*skeleton.bones[bones_in.y]+weights.z*skeleton.bones[bones_in.z]+weights.w*skeleton.bones[bones_in.w]; vec4 p=skin*vec4(position,1); world_position=p.xyz; gl_Position=p; }
