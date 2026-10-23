#version 450
layout(set=0,binding=0) uniform sampler2D albedo_map; layout(set=0,binding=1) uniform sampler2D normal_map;
layout(location=0) in vec2 uv; layout(location=1) in vec3 light_dir; layout(location=0) out vec4 color;
void main(){ vec3 albedo=texture(albedo_map,uv).rgb; vec3 n=texture(normal_map,uv).xyz*2.0-1.0; float d=max(dot(normalize(n),normalize(light_dir)),0.0); color=vec4(albedo*d,1.0); }
