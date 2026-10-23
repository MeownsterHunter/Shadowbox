#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(location=0) in f16vec3 input_color; layout(location=0) out vec4 color;
void main(){ f16vec3 adjusted=input_color*f16vec3(0.75hf)+f16vec3(0.1hf); color=vec4(adjusted,1); }
