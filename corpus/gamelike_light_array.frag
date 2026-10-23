#version 450
struct Light { vec4 position; vec4 color; }; layout(set=0,binding=0) uniform Lights { Light lights[4]; } scene;
layout(location=0) in vec3 position; layout(location=1) in vec3 nrm; layout(location=0) out vec4 color;
void main(){ vec3 total=vec3(0.0); for(int i=0;i<4;i++){ vec3 l=normalize(scene.lights[i].position.xyz+position); total+=scene.lights[i].color.rgb*max(dot(normalize(nrm),l),0.0); } color=vec4(total,1.0); }
