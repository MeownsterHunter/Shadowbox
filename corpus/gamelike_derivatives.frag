#version 450
layout(location=0) in vec2 uv; layout(location=0) out vec4 color;
void main(){ vec2 dx=dFdx(uv), dy=dFdy(uv); float footprint=length(dx)+length(dy); color=vec4(dx,dy.x+footprint,1.0); }
