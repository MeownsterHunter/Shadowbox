#version 450
layout(set=0,binding=0) uniform sampler2D base_map; layout(location=0) in vec2 uv; layout(location=0) out vec4 color;
void main(){ vec4 c=texture(base_map,uv); if(c.a<0.5) discard; color=c; }
