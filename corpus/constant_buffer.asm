block.u 0 L0 [] -> {
 r0 = mov c[0x1][0x0] // delay=1
 r1 = mov c[0x1][0x4] // delay=1
 r2 = mov c[0x1][0x8] // delay=1
 r3 = mov c[0x1][0xc] // delay=1
      exit            // delay=13
      nop             // delay=2
} -> []
