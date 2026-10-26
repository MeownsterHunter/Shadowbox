block.u 0 L0 [] -> {
 r0 = ipa.pass a[0x7c] rZ       // delay=6 wr:0
 r0 = mufu.rcp.f32 r0           // delay=6 wt=000001 wr:0
 r0 = ipa.pass_mul_w a[0x80] r0 // delay=6 wt=000001 wr:0
 r0 = f2f.ftz.int.f32.f32.rz r0 // delay=6 wt=000001 wr:0
 r3 = mov r0                    // delay=1 wt=000001
 r2 = mov r0                    // delay=1
 r1 = mov r0                    // delay=1
      exit                      // delay=13
      nop                       // delay=2
} -> []
