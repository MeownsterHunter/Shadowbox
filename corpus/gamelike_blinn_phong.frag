#version 450
layout(location=0) in vec3 nrm; layout(location=1) in vec3 view_dir; layout(location=2) in vec3 light_dir;
layout(location=0) out vec4 color; layout(set=0,binding=0) uniform M { vec4 diffuse; vec4 specular; } mat;
void main(){ vec3 n=normalize(nrm), l=normalize(light_dir), v=normalize(view_dir); float d=max(dot(n,l),0.0); float s=pow(max(dot(reflect(-l,n),v),0.0),16.0); vec3 lit=mat.diffuse.rgb*d+mat.specular.rgb*s; color=vec4(mix(mat.diffuse.rgb,lit,0.8),mat.diffuse.a); }
