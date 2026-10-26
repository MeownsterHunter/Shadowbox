#version 450
layout(location=0) in vec4 input_color; layout(location=1) in vec3 nrm; layout(location=0) out vec4 albedo; layout(location=1) out vec4 normal_target; layout(location=2) out vec4 material;
void main(){ albedo=input_color; normal_target=vec4(normalize(nrm)*0.5+0.5,1.0); material=vec4(input_color.a,dot(normalize(nrm),vec3(0,0,1)),0,1); }
