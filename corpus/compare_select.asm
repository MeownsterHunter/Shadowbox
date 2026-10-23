block.u 0 L0 [] -> {
 r1 = ipa.pass a[0x7c] rZ        // delay=6 wr:0
 r1 = mufu.rcp.f32 r1            // delay=6 wt=000001 wr:0
 r1 = ipa.pass_mul_w a[0x80] r1  // delay=6 wt=000001 wr:0
 p0 = fsetp.lt.ftz r1 0x3f000000 // delay=7 wt=000001
 r0 = mov 0x3f800000             // delay=6
 r0 = sel !p0 rZ r0              // delay=6
 r3 = mov r0                     // delay=1
 r2 = mov r0                     // delay=1
 r1 = mov r0                     // delay=1
      exit                       // delay=13
      nop                        // delay=2
} -> []
