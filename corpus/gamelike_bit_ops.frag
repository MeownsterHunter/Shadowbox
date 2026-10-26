#version 450
layout(location=0) flat in uint flags; layout(location=0) out vec4 color;
void main(){ uint masked=flags&0xffu; uint shifted=(masked<<2u)|(flags>>6u); uint mixed=shifted^0x5au; color=vec4(float(mixed&0xffu)/255.0,float(bitCount(mixed))/32.0,0,1); }
