block.u 0 L0 [] -> {
 r0 = ipa.constant a[0x80] rZ // delay=6 wr:0
 r1 = shl.w r0 0x2            // delay=4 wt=000001
 r0 = shr.w.u32 r0 0x1        // delay=6
 r0 = lop2.xor r1 r0          // delay=6
 r0 = prmt r0 [0x4440] rZ     // delay=6
 r0 = i2f.f32.u32.re r0       // delay=6 wr:0
 r0 = fmul.ftz r0 0x3b808081  // delay=6 wt=000001
 r3 = mov r0                  // delay=1
 r2 = mov r0                  // delay=1
 r1 = mov r0                  // delay=1
      exit                    // delay=13
      nop                     // delay=2
} -> []
