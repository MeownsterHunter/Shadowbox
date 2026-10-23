block.u 0 L0 [] -> {
 r0 = mov 0x3f800000 // delay=4
 r1 = mov 0x3e800000 // delay=1
 r2 = mov 0x3f000000 // delay=1
 r3 = mov r0         // delay=1
      exit           // delay=13
      nop            // delay=2
} -> []
